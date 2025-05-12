//! The Forge test runner.

use crate::{
    fuzz::{invariant::BasicTxDetails, BaseCounterExample},
    multi_runner::{is_matching_test, TestContract, TestRunnerConfig},
    progress::{start_fuzz_progress, TestsProgress},
    result::{SuiteResult, TestResult, TestSetup},
    MultiContractRunner, TestFilter,
};
use alloy_dyn_abi::DynSolValue;
use alloy_json_abi::Function;
use alloy_primitives::{address, map::HashMap, Address, Bytes, U256};
use alloy_sol_macro::sol;
use eyre::Result;
use foundry_common::{contracts::ContractsByAddress, TestFunctionExt, TestFunctionKind};
use foundry_compilers::utils::canonicalized;
use foundry_config::{Config, InvariantConfig};
use foundry_evm::{
    constants::{CALLER, CHEATCODE_ADDRESS},
    decode::RevertDecoder,
    executors::{
        fuzz::FuzzedExecutor,
        invariant::{
            check_sequence, replay_error, replay_run, InvariantExecutor, InvariantFuzzError,
        },
        CallResult, EvmError, Executor, ITest, RawCallResult, ConfigTestDiff
    },
    fuzz::{
        fixture_name,
        invariant::{CallDetails, InvariantContract},
        CounterExample, FuzzFixtures,
    },
    traces::{load_contracts, TraceKind, TraceMode},
};
use proptest::test_runner::{
    FailurePersistence, FileFailurePersistence, RngAlgorithm, TestError, TestRng, TestRunner,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    cmp::min,
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::Span;

/// When running tests, we deploy all external libraries present in the project. To avoid additional
/// libraries affecting nonces of senders used in tests, we are using separate address to
/// predeploy libraries.
///
/// `address(uint160(uint256(keccak256("foundry library deployer"))))`
pub const LIBRARY_DEPLOYER: Address = address!("0x1F95D37F27EA0dEA9C252FC09D5A6eaA97647353");

/// A type that executes all tests of a contract
pub struct ContractRunner<'a> {
    /// The name of the contract.
    name: &'a str,
    /// The data of the contract.
    contract: &'a TestContract,
    /// The EVM executor.
    executor: Executor,
    /// Overall test run progress.
    progress: Option<&'a TestsProgress>,
    /// The handle to the tokio runtime.
    tokio_handle: &'a tokio::runtime::Handle,
    /// The span of the contract.
    span: tracing::Span,
    /// The contract-level configuration.
    tcfg: Cow<'a, TestRunnerConfig>,
    /// The parent runner.
    mcr: &'a MultiContractRunner,
}

impl<'a> std::ops::Deref for ContractRunner<'a> {
    type Target = Cow<'a, TestRunnerConfig>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.tcfg
    }
}

impl<'a> ContractRunner<'a> {
    pub fn new(
        name: &'a str,
        contract: &'a TestContract,
        executor: Executor,
        progress: Option<&'a TestsProgress>,
        tokio_handle: &'a tokio::runtime::Handle,
        span: Span,
        mcr: &'a MultiContractRunner,
    ) -> Self {
        Self {
            name,
            contract,
            executor,
            progress,
            tokio_handle,
            span,
            tcfg: Cow::Borrowed(&mcr.tcfg),
            mcr,
        }
    }

    /// Deploys the test contract inside the runner from the sending account, and optionally runs
    /// the `setUp` function on the test contract.
    pub fn setup(&mut self, call_setup: bool) -> TestSetup {
        self._setup(call_setup).unwrap_or_else(|err| {
            if err.to_string().contains("skipped") {
                TestSetup::skipped(err.to_string())
            } else {
                TestSetup::failed(err.to_string())
            }
        })
    }

    fn _setup(&mut self, call_setup: bool) -> Result<TestSetup> {
        trace!(call_setup, "setting up");

        self.apply_contract_inline_config()?;

        // We max out their balance so that they can deploy and make calls.
        self.executor.set_balance(self.sender, U256::MAX)?;
        self.executor.set_balance(CALLER, U256::MAX)?;

        // We set the nonce of the deployer accounts to 1 to get the same addresses as DappTools.
        self.executor.set_nonce(self.sender, 1)?;

        // Deploy libraries.
        self.executor.set_balance(LIBRARY_DEPLOYER, U256::MAX)?;

        let mut result = TestSetup::default();
        for code in &self.mcr.libs_to_deploy {
            let deploy_result = self.executor.deploy(
                LIBRARY_DEPLOYER,
                code.clone(),
                U256::ZERO,
                Some(&self.mcr.revert_decoder),
            );

            // Record deployed library address.
            if let Ok(deployed) = &deploy_result {
                result.deployed_libs.push(deployed.address);
            }

            let (raw, reason) = RawCallResult::from_evm_result(deploy_result.map(Into::into))?;
            result.extend(raw, TraceKind::Deployment);
            if reason.is_some() {
                result.reason = reason;
                return Ok(result);
            }
        }

        let address = self.sender.create(self.executor.get_nonce(self.sender)?);
        result.address = address;

        // Set the contracts initial balance before deployment, so it is available during
        // construction
        self.executor.set_balance(address, self.initial_balance())?;

        // Deploy the test contract
        let deploy_result = self.executor.deploy(
            self.sender,
            self.contract.bytecode.clone(),
            U256::ZERO,
            Some(&self.mcr.revert_decoder),
        );

        result.deployment_failure = deploy_result.is_err();

        if let Ok(dr) = &deploy_result {
            debug_assert_eq!(dr.address, address);
        }
        let (raw, reason) = RawCallResult::from_evm_result(deploy_result.map(Into::into))?;
        result.extend(raw, TraceKind::Deployment);
        if reason.is_some() {
            result.reason = reason;
            return Ok(result);
        }

        // Reset `self.sender`s, `CALLER`s and `LIBRARY_DEPLOYER`'s balance to the initial balance.
        self.executor.set_balance(self.sender, self.initial_balance())?;
        self.executor.set_balance(CALLER, self.initial_balance())?;
        self.executor.set_balance(LIBRARY_DEPLOYER, self.initial_balance())?;

        self.executor.deploy_create2_deployer()?;

        // Optionally call the `setUp` function
        if call_setup {
            trace!("calling setUp");
            let res = self.executor.setup(None, address, Some(&self.mcr.revert_decoder));
            let (raw, reason) = RawCallResult::from_evm_result(res)?;
            result.extend(raw, TraceKind::Setup);
            result.reason = reason;
        }

        result.fuzz_fixtures = self.fuzz_fixtures(address);

        Ok(result)
    }

    fn initial_balance(&self) -> U256 {
        self.evm_opts.initial_balance
    }

    /// Configures this runner with the inline configuration for the contract.
    fn apply_contract_inline_config(&mut self) -> Result<()> {
        if self.inline_config.contains_contract(self.name) {
            let new_config = Arc::new(self.inline_config(None)?);
            self.tcfg.to_mut().reconfigure_with(new_config);
            let prev_tracer = self.executor.inspector_mut().tracer.take();
            self.tcfg.configure_executor(&mut self.executor);
            // Don't set tracer here.
            self.executor.inspector_mut().tracer = prev_tracer;
        }
        Ok(())
    }

    /// Returns the configuration for a contract or function.
    fn inline_config(&self, func: Option<&Function>) -> Result<Config> {
        let function = func.map(|f| f.name.as_str()).unwrap_or("");
        let config =
            self.mcr.inline_config.merge(self.name, function, &self.config).extract::<Config>()?;
        Ok(config)
    }

    /// Collect fixtures from test contract.
    ///
    /// Fixtures can be defined:
    /// - as storage arrays in test contract, prefixed with `fixture`
    /// - as functions prefixed with `fixture` and followed by parameter name to be fuzzed
    ///
    /// Storage array fixtures:
    /// `uint256[] public fixture_amount = [1, 2, 3];`
    /// define an array of uint256 values to be used for fuzzing `amount` named parameter in scope
    /// of the current test.
    ///
    /// Function fixtures:
    /// `function fixture_owner() public returns (address[] memory){}`
    /// returns an array of addresses to be used for fuzzing `owner` named parameter in scope of the
    /// current test.
    fn fuzz_fixtures(&mut self, address: Address) -> FuzzFixtures {
        let mut fixtures = HashMap::default();
        let fixture_functions = self.contract.abi.functions().filter(|func| func.is_fixture());
        for func in fixture_functions {
            if func.inputs.is_empty() {
                // Read fixtures declared as functions.
                if let Ok(CallResult { raw: _, decoded_result }) =
                    self.executor.call(CALLER, address, func, &[], U256::ZERO, None)
                {
                    fixtures.insert(fixture_name(func.name.clone()), decoded_result);
                }
            } else {
                // For reading fixtures from storage arrays we collect values by calling the
                // function with incremented indexes until there's an error.
                let mut vals = Vec::new();
                let mut index = 0;
                loop {
                    if let Ok(CallResult { raw: _, decoded_result }) = self.executor.call(
                        CALLER,
                        address,
                        func,
                        &[DynSolValue::Uint(U256::from(index), 256)],
                        U256::ZERO,
                        None,
                    ) {
                        vals.push(decoded_result);
                    } else {
                        // No result returned for this index, we reached the end of storage
                        // array or the function is not a valid fixture.
                        break;
                    }
                    index += 1;
                }
                fixtures.insert(fixture_name(func.name.clone()), DynSolValue::Array(vals));
            };
        }
        FuzzFixtures::new(fixtures)
    }

    /// Runs all tests for a contract whose names match the provided regular expression
    pub fn run_tests(mut self, filter: &dyn TestFilter) -> SuiteResult {
        let start = Instant::now();
        let mut warnings = Vec::new();

        // Check if `setUp` function with valid signature declared.
        let setup_fns: Vec<_> =
            self.contract.abi.functions().filter(|func| func.name.is_setup()).collect();
        let call_setup = setup_fns.len() == 1 && setup_fns[0].name == "setUp";
        // There is a single miss-cased `setUp` function, so we add a warning
        for &setup_fn in &setup_fns {
            if setup_fn.name != "setUp" {
                warnings.push(format!(
                    "Found invalid setup function \"{}\" did you mean \"setUp()\"?",
                    setup_fn.signature()
                ));
            }
        }

        // There are multiple setUp function, so we return a single test result for `setUp`
        if setup_fns.len() > 1 {
            return SuiteResult::new(
                start.elapsed(),
                [("setUp()".to_string(), TestResult::fail("multiple setUp functions".to_string()))]
                    .into(),
                warnings,
            )
        }

        // Check if `afterInvariant` function with valid signature declared.
        let after_invariant_fns: Vec<_> =
            self.contract.abi.functions().filter(|func| func.name.is_after_invariant()).collect();
        if after_invariant_fns.len() > 1 {
            // Return a single test result failure if multiple functions declared.
            return SuiteResult::new(
                start.elapsed(),
                [(
                    "afterInvariant()".to_string(),
                    TestResult::fail("multiple afterInvariant functions".to_string()),
                )]
                .into(),
                warnings,
            )
        }
        let call_after_invariant = after_invariant_fns.first().is_some_and(|after_invariant_fn| {
            let match_sig = after_invariant_fn.name == "afterInvariant";
            if !match_sig {
                warnings.push(format!(
                    "Found invalid afterInvariant function \"{}\" did you mean \"afterInvariant()\"?",
                    after_invariant_fn.signature()
                ));
            }
            match_sig
        });

        // Invariant testing requires tracing to figure out what contracts were created.
        // We also want to disable `debug` for setup since we won't be using those traces.
        let has_invariants = self.contract.abi.functions().any(|func| func.is_invariant_test());

        let prev_tracer = self.executor.inspector_mut().tracer.take();
        if prev_tracer.is_some() || has_invariants {
            self.executor.set_tracing(TraceMode::Call);
        }

        let setup_time = Instant::now();
        let setup = self.setup(call_setup);
        debug!("finished setting up in {:?}", setup_time.elapsed());

        self.executor.inspector_mut().tracer = prev_tracer;

        if setup.reason.is_some() {
            // The setup failed, so we return a single test result for `setUp`
            let fail_msg = if !setup.deployment_failure {
                "setUp()".to_string()
            } else {
                "constructor()".to_string()
            };
            return SuiteResult::new(
                start.elapsed(),
                [(fail_msg, TestResult::setup_result(setup))].into(),
                warnings,
            )
        }

        // Filter out functions sequentially since it's very fast and there is no need to do it
        // in parallel.
        let find_timer = Instant::now();
        let functions = self
            .contract
            .abi
            .functions()
            .filter(|func| is_matching_test(func, filter))
            .collect::<Vec<_>>();
        debug!(
            "Found {} test functions out of {} in {:?}",
            functions.len(),
            self.contract.abi.functions().count(),
            find_timer.elapsed(),
        );

        let identified_contracts = has_invariants.then(|| {
            load_contracts(setup.traces.iter().map(|(_, t)| &t.arena), &self.mcr.known_contracts)
        });

        let test_fail_instances = functions
            .iter()
            .filter_map(|func| {
                TestFunctionKind::classify(&func.name, !func.inputs.is_empty())
                    .is_any_test_fail()
                    .then_some(func.name.clone())
            })
            .collect::<Vec<_>>();

        if !test_fail_instances.is_empty() {
            let instances = format!(
                "Found {} instances: {}",
                test_fail_instances.len(),
                test_fail_instances.join(", ")
            );
            let fail =  TestResult::fail("`testFail*` has been removed. Consider changing to test_Revert[If|When]_Condition and expecting a revert".to_string());
            return SuiteResult::new(start.elapsed(), [(instances, fail)].into(), warnings)
        }

        let test_results = functions
            .par_iter()
            .map(|&func| {
                let start = Instant::now();

                let _guard = self.tokio_handle.enter();

                let _guard;
                let current_span = tracing::Span::current();
                if current_span.is_none() || current_span.id() != self.span.id() {
                    _guard = self.span.enter();
                }

                let sig = func.signature();
                let kind = func.test_function_kind();

                let _guard = debug_span!(
                    "test",
                    %kind,
                    name = %if enabled!(tracing::Level::TRACE) { &sig } else { &func.name },
                )
                .entered();

                let mut res = FunctionRunner::new(&self, &setup).run(
                    func,
                    kind,
                    call_after_invariant,
                    identified_contracts.as_ref(),
                );
                res.duration = start.elapsed();

                (sig, res)
            })
            .collect::<BTreeMap<_, _>>();

        let duration = start.elapsed();
        SuiteResult::new(duration, test_results, warnings)
    }
}

sol! {
    #[derive(Debug)]
    struct ChainInfo {
        uint256 forkId;
        uint256 chainId;
    }

    #[derive(Debug)]
    enum AccountAccessKind {
        Call,
        DelegateCall,
        CallCode,
        StaticCall,
        Create,
        SelfDestruct,
        Resume,
        Balance,
        Extcodesize,
        Extcodehash,
        Extcodecopy,
    }

    #[derive(Debug)]
    struct StorageAccess {
        address account;
        bytes32 slot;
        bool isWrite;
        bytes32 previousValue;
        bytes32 newValue;
        bool reverted;
    }

    #[derive(Debug)]
    struct AccountAccess {
        ChainInfo chainInfo;
        AccountAccessKind kind;
        address account;
        address accessor;
        bool initialized;
        uint256 oldBalance;
        uint256 newBalance;
        bytes deployedCode;
        uint256 value;
        bytes data;
        bool reverted;
        StorageAccess[] storageAccesses;
        uint64 depth;
    }
}

/// Executes a single test function, returning a [`TestResult`].
struct FunctionRunner<'a> {
    /// The function-level configuration.
    tcfg: Cow<'a, TestRunnerConfig>,
    /// The EVM executor.
    executor: Cow<'a, Executor>,
    /// The parent runner.
    cr: &'a ContractRunner<'a>,
    /// The address of the test contract.
    address: Address,
    /// The test setup result.
    setup: &'a TestSetup,
    /// The test result. Returned after running the test.
    result: TestResult,
}

impl<'a> std::ops::Deref for FunctionRunner<'a> {
    type Target = Cow<'a, TestRunnerConfig>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.tcfg
    }
}

impl<'a> FunctionRunner<'a> {
    fn new(cr: &'a ContractRunner<'a>, setup: &'a TestSetup) -> Self {
        Self {
            tcfg: match &cr.tcfg {
                Cow::Borrowed(tcfg) => Cow::Borrowed(tcfg),
                Cow::Owned(tcfg) => Cow::Owned(tcfg.clone()),
            },
            executor: Cow::Borrowed(&cr.executor),
            cr,
            address: setup.address,
            setup,
            result: TestResult::new(setup),
        }
    }

    fn revert_decoder(&self) -> &'a RevertDecoder {
        &self.cr.mcr.revert_decoder
    }

    /// Configures this runner with the inline configuration for the contract.
    fn apply_function_inline_config(&mut self, func: &Function) -> Result<()> {
        if self.inline_config.contains_function(self.cr.name, &func.name) {
            let new_config = Arc::new(self.cr.inline_config(Some(func))?);
            self.tcfg.to_mut().reconfigure_with(new_config);
            self.tcfg.configure_executor(self.executor.to_mut());
        }
        Ok(())
    }

    fn run(
        mut self,
        func: &Function,
        kind: TestFunctionKind,
        call_after_invariant: bool,
        identified_contracts: Option<&ContractsByAddress>,
    ) -> TestResult {
        if let Err(e) = self.apply_function_inline_config(func) {
            self.result.single_fail(Some(e.to_string()));
            return self.result;
        }

        match kind {
            TestFunctionKind::UnitTest { .. } => self.run_unit_test(func),
            TestFunctionKind::FuzzTest { .. } => self.run_fuzz_test(func),
            TestFunctionKind::InvariantTest => {
                let test_bytecode = &self.cr.contract.bytecode;
                self.run_invariant_test(
                    func,
                    call_after_invariant,
                    identified_contracts.unwrap(),
                    test_bytecode,
                )
            }
            TestFunctionKind::DifferentialTest => self.run_differential_test(func),
            _ => unreachable!(),
        }
    }

    /// Runs a single unit test.
    ///
    /// Applies before test txes (if any), runs current test and returns the `TestResult`.
    ///
    /// Before test txes are applied in order and state modifications committed to the EVM database
    /// (therefore the unit test call will be made on modified state).
    /// State modifications of before test txes and unit test function call are discarded after
    /// test ends, similar to `eth_call`.
    fn run_unit_test(mut self, func: &Function) -> TestResult {
        // Prepare unit test execution.
        if self.prepare_test(func).is_err() {
            return self.result;
        }

        // Run current unit test.
        let (mut raw_call_result, reason) = match self.executor.call(
            self.sender,
            self.address,
            func,
            &[],
            U256::ZERO,
            Some(self.revert_decoder()),
        ) {
            Ok(res) => (res.raw, None),
            Err(EvmError::Execution(err)) => (err.raw, Some(err.reason)),
            Err(EvmError::Skip(reason)) => {
                self.result.single_skip(reason);
                return self.result;
            }
            Err(err) => {
                self.result.single_fail(Some(err.to_string()));
                return self.result;
            }
        };

        let success =
            self.executor.is_raw_call_mut_success(self.address, &mut raw_call_result, false);
        self.result.single_result(success, reason, raw_call_result);
        self.result
    }

    fn run_invariant_test(
        mut self,
        func: &Function,
        call_after_invariant: bool,
        identified_contracts: &ContractsByAddress,
        test_bytecode: &Bytes,
    ) -> TestResult {
        // First, run the test normally to see if it needs to be skipped.
        if let Err(EvmError::Skip(reason)) = self.executor.call(
            self.sender,
            self.address,
            func,
            &[],
            U256::ZERO,
            Some(self.revert_decoder()),
        ) {
            self.result.invariant_skip(reason);
            return self.result;
        };

        let runner = self.invariant_runner();
        let invariant_config = &self.config.invariant;

        let mut evm = InvariantExecutor::new(
            self.clone_executor(),
            runner,
            invariant_config.clone(),
            identified_contracts,
            &self.cr.mcr.known_contracts,
        );
        let invariant_contract = InvariantContract {
            address: self.address,
            invariant_function: func,
            call_after_invariant,
            abi: &self.cr.contract.abi,
        };

        let (failure_dir, failure_file) = invariant_failure_paths(
            invariant_config,
            self.cr.name,
            &invariant_contract.invariant_function.name,
        );
        let show_solidity = invariant_config.show_solidity;

        // Try to replay recorded failure if any.
        if let Some(mut call_sequence) =
            persisted_call_sequence(failure_file.as_path(), test_bytecode)
        {
            // Create calls from failed sequence and check if invariant still broken.
            let txes = call_sequence
                .iter_mut()
                .map(|seq| {
                    seq.show_solidity = show_solidity;
                    BasicTxDetails {
                        sender: seq.sender.unwrap_or_default(),
                        call_details: CallDetails {
                            target: seq.addr.unwrap_or_default(),
                            calldata: seq.calldata.clone(),
                        },
                    }
                })
                .collect::<Vec<BasicTxDetails>>();
            if let Ok((success, replayed_entirely)) = check_sequence(
                self.clone_executor(),
                &txes,
                (0..min(txes.len(), invariant_config.depth as usize)).collect(),
                invariant_contract.address,
                invariant_contract.invariant_function.selector().to_vec().into(),
                invariant_config.fail_on_revert,
                invariant_contract.call_after_invariant,
            ) {
                if !success {
                    let _= sh_warn!("\
                            Replayed invariant failure from {:?} file. \
                            Run `forge clean` or remove file to ignore failure and to continue invariant test campaign.",
                        failure_file.as_path()
                    );
                    // If sequence still fails then replay error to collect traces and
                    // exit without executing new runs.
                    let _ = replay_run(
                        &invariant_contract,
                        self.clone_executor(),
                        &self.cr.mcr.known_contracts,
                        identified_contracts.clone(),
                        &mut self.result.logs,
                        &mut self.result.traces,
                        &mut self.result.coverage,
                        &mut self.result.deprecated_cheatcodes,
                        &txes,
                        show_solidity,
                    );
                    self.result.invariant_replay_fail(
                        replayed_entirely,
                        &invariant_contract.invariant_function.name,
                        call_sequence,
                    );
                    return self.result;
                }
            }
        }

        let progress =
            start_fuzz_progress(self.cr.progress, self.cr.name, &func.name, invariant_config.runs);
        let invariant_result = match evm.invariant_fuzz(
            invariant_contract.clone(),
            &self.setup.fuzz_fixtures,
            &self.setup.deployed_libs,
            progress.as_ref(),
        ) {
            Ok(x) => x,
            Err(e) => {
                self.result.invariant_setup_fail(e);
                return self.result;
            }
        };
        // Merge coverage collected during invariant run with test setup coverage.
        self.result.merge_coverages(invariant_result.coverage);

        let mut counterexample = None;
        let success = invariant_result.error.is_none();
        let reason = invariant_result.error.as_ref().and_then(|err| err.revert_reason());

        match invariant_result.error {
            // If invariants were broken, replay the error to collect logs and traces
            Some(error) => match error {
                InvariantFuzzError::BrokenInvariant(case_data) |
                InvariantFuzzError::Revert(case_data) => {
                    // Replay error to create counterexample and to collect logs, traces and
                    // coverage.
                    match replay_error(
                        &case_data,
                        &invariant_contract,
                        self.clone_executor(),
                        &self.cr.mcr.known_contracts,
                        identified_contracts.clone(),
                        &mut self.result.logs,
                        &mut self.result.traces,
                        &mut self.result.coverage,
                        &mut self.result.deprecated_cheatcodes,
                        progress.as_ref(),
                        show_solidity,
                    ) {
                        Ok(call_sequence) => {
                            if !call_sequence.is_empty() {
                                // Persist error in invariant failure dir.
                                if let Err(err) = foundry_common::fs::create_dir_all(failure_dir) {
                                    error!(%err, "Failed to create invariant failure dir");
                                } else if let Err(err) = foundry_common::fs::write_json_file(
                                    failure_file.as_path(),
                                    &InvariantPersistedFailure {
                                        call_sequence: call_sequence.clone(),
                                        driver_bytecode: Some(test_bytecode.clone()),
                                    },
                                ) {
                                    error!(%err, "Failed to record call sequence");
                                }

                                let original_seq_len =
                                    if let TestError::Fail(_, calls) = &case_data.test_error {
                                        calls.len()
                                    } else {
                                        call_sequence.len()
                                    };

                                counterexample =
                                    Some(CounterExample::Sequence(original_seq_len, call_sequence))
                            }
                        }
                        Err(err) => {
                            error!(%err, "Failed to replay invariant error");
                        }
                    };
                }
                InvariantFuzzError::MaxAssumeRejects(_) => {}
            },

            // If invariants ran successfully, replay the last run to collect logs and
            // traces.
            _ => {
                if let Err(err) = replay_run(
                    &invariant_contract,
                    self.clone_executor(),
                    &self.cr.mcr.known_contracts,
                    identified_contracts.clone(),
                    &mut self.result.logs,
                    &mut self.result.traces,
                    &mut self.result.coverage,
                    &mut self.result.deprecated_cheatcodes,
                    &invariant_result.last_run_inputs,
                    show_solidity,
                ) {
                    error!(%err, "Failed to replay last invariant run");
                }
            }
        }

        self.result.invariant_result(
            invariant_result.gas_report_traces,
            success,
            reason,
            counterexample,
            invariant_result.cases,
            invariant_result.reverts,
            invariant_result.metrics,
        );
        self.result
    }

    /// Runs a fuzzed test.
    ///
    /// Applies the before test txes (if any), fuzzes the current function and returns the
    /// `TestResult`.
    ///
    /// Before test txes are applied in order and state modifications committed to the EVM database
    /// (therefore the fuzz test will use the modified state).
    /// State modifications of before test txes and fuzz test are discarded after test ends,
    /// similar to `eth_call`.
    fn run_fuzz_test(mut self, func: &Function) -> TestResult {
        // Prepare fuzz test execution.
        if self.prepare_test(func).is_err() {
            return self.result;
        }

        let runner = self.fuzz_runner();
        let fuzz_config = self.config.fuzz.clone();

        let progress =
            start_fuzz_progress(self.cr.progress, self.cr.name, &func.name, fuzz_config.runs);

        // Run fuzz test.
        let fuzzed_executor =
            FuzzedExecutor::new(self.executor.into_owned(), runner, self.tcfg.sender, fuzz_config);
        let result = fuzzed_executor.fuzz(
            func,
            &self.setup.fuzz_fixtures,
            &self.setup.deployed_libs,
            self.address,
            &self.cr.mcr.revert_decoder,
            progress.as_ref(),
        );
        self.result.fuzz_result(result);
        self.result
    }

    /// Prepares single unit test and fuzz test execution:
    /// - set up the test result and executor
    /// - check if before test txes are configured and apply them in order
    ///
    /// Before test txes are arrays of arbitrary calldata obtained by calling the `beforeTest`
    /// function with test selector as a parameter.
    ///
    /// Unit tests within same contract (or even current test) are valid options for before test tx
    /// configuration. Test execution stops if any of before test txes fails.
    fn prepare_test(&mut self, func: &Function) -> Result<(), ()> {
        let address = self.setup.address;

        // Apply before test configured functions (if any).
        if self.cr.contract.abi.functions().filter(|func| func.name.is_before_test_setup()).count() ==
            1
        {
            for calldata in self
                .executor
                .call_sol_default(
                    address,
                    &ITest::beforeTestSetupCall { testSelector: func.selector() },
                )
                .beforeTestCalldata
            {
                // Apply before test configured calldata.
                match self.executor.to_mut().transact_raw(
                    self.tcfg.sender,
                    address,
                    calldata,
                    U256::ZERO,
                ) {
                    Ok(call_result) => {
                        let reverted = call_result.reverted;

                        // Merge tx result traces in unit test result.
                        self.result.extend(call_result);

                        // To continue unit test execution the call should not revert.
                        if reverted {
                            self.result.single_fail(None);
                            return Err(());
                        }
                    }
                    Err(_) => {
                        self.result.single_fail(None);
                        return Err(());
                    }
                }
            }
        }
        Ok(())
    }

    fn fuzz_runner(&self) -> TestRunner {
        let config = &self.config.fuzz;
        let failure_persist_path = config
            .failure_persist_dir
            .as_ref()
            .unwrap()
            .join(config.failure_persist_file.as_ref().unwrap())
            .into_os_string()
            .into_string()
            .unwrap();
        fuzzer_with_cases(
            config.seed,
            config.runs,
            config.max_test_rejects,
            Some(Box::new(FileFailurePersistence::Direct(failure_persist_path.leak()))),
        )
    }

    fn invariant_runner(&self) -> TestRunner {
        let config = &self.config.invariant;
        fuzzer_with_cases(self.config.fuzz.seed, config.runs, config.max_assume_rejects, None)
    }

    fn clone_executor(&self) -> Executor {
        self.executor.clone().into_owned()
    }

    /// Runs a differential test.
    /// This type of test compares the execution results between different implementations
    fn run_differential_test(mut self, func: &Function) -> TestResult {
        // Prepare differential test execution
        if self.prepare_test(func).is_err() {
            return self.result;
        }

        // Get the test configuration by calling the test function
        let revert_decoder = self.revert_decoder();
        let (get_config_raw_call_result, reason) = match self.executor.to_mut().transact(
            self.tcfg.sender,
            self.address,
            func,
            &[],
            U256::ZERO,
            Some(&revert_decoder),
        ) {
            Ok(res) => (res.raw, None),
            Err(EvmError::Execution(err)) => (err.raw, Some(err.reason)),
            Err(EvmError::Skip(reason)) => {
                self.result.single_skip(reason);
                return self.result;
            }
            Err(err) => {
                self.result.single_fail(Some(err.to_string()));
                return self.result;
            }
        };

        // Try to decode the ConfigTestDiff from the result
        let config = match ConfigTestDiff::decode_from_bytes(&get_config_raw_call_result.result) {
            Ok(config) => config,
            Err(err) => {
                self.result.single_fail(Some(format!("Failed to decode ConfigTestDiff: {}", err)));
                return self.result;
            }
        };

        // TODO: исправить чтоб ошибки выводились, сейчас - не работает!
        // Saves the current state of the EVM with 'vm.snapshotState()'
        let snapshot_calldata = Bytes::from_static(&[0x9c, 0xd2, 0x38, 0x35]);
        let snapshot_result = match self.executor.to_mut().transact_raw(self.tcfg.sender, CHEATCODE_ADDRESS, snapshot_calldata, U256::ZERO) {
            Ok(result) => result,
            Err(err) => {
                self.result.single_fail(Some(format!("Snapshot failed: {}", err)));
                return self.result;
            }
        };
        let snapshot_id = Bytes::copy_from_slice(&snapshot_result.result);

        // Starts recording the EVM state changes with 'vm.startRecording()'
        let start_recording_calldata = Bytes::from_static(&[0xcf, 0x22, 0xe3, 0xc9]);
        if let Err(err) = self.executor.to_mut().transact_raw(self.tcfg.sender, CHEATCODE_ADDRESS, start_recording_calldata, U256::ZERO) {
            self.result.single_fail(Some(format!("Start recording failed: {}", err)));
            return self.result;
        }

        // Execute first function call
        let first_call = match self.executor.to_mut().transact_raw(
            config.from_for_fn1,
            config.addr_contract_with_fn1,
            config.calldata_fn1,
            U256::ZERO,
        ) {
            Ok(result) => result,
            Err(err) => {
                self.result.single_fail(Some(format!("First function call failed: {}", err)));
                return self.result;
            }
        };

        // Stops recording the EVM state changes with 'vm.stopAndReturnStateDiff()'
        let stop_recording_calldata = Bytes::from_static(&[0xaa, 0x5c, 0xf9, 0x0e]);
        let first_diff_result = match self.executor.to_mut().transact_raw(self.tcfg.sender, CHEATCODE_ADDRESS, stop_recording_calldata, U256::ZERO) {
            Ok(result) => result,
            Err(err) => {
                self.result.single_fail(Some(format!("Stop recording failed: {}", err)));
                return self.result;
            }
        };

        // Decode the state diff result
        let first_diff: Vec<AccountAccess> = match alloy_sol_types::SolValue::abi_decode(&first_diff_result.result, false) {
            Ok(diff) => diff,
            Err(e) => {
                self.result.single_fail(Some(format!("Failed to decode first state diff: {}", e)));
                return self.result;
            }
        };

        // eprintln!("first_state_diffs: {:?}", first_diff);

        let first_state_changes = process_state_diffs(&first_diff, config.addr_contract_with_fn1);

        // Reverts the EVM state changes with 'vm.revertToState()'
        let revert_calldata = Bytes::from_iter([&[0xc2, 0x52, 0x74, 0x05][..], snapshot_id.as_ref()].concat());
        if let Err(err) = self.executor.to_mut().transact_raw(self.tcfg.sender, CHEATCODE_ADDRESS, revert_calldata, U256::ZERO) {
            self.result.single_fail(Some(format!("Revert failed: {}", err)));
            return self.result;
        }

        // Starts recording the EVM state changes with 'vm.startRecording()'
        let start_recording_calldata = Bytes::from_static(&[0xcf, 0x22, 0xe3, 0xc9]);
        if let Err(err) = self.executor.to_mut().transact_raw(self.tcfg.sender, CHEATCODE_ADDRESS, start_recording_calldata, U256::ZERO) {
            self.result.single_fail(Some(format!("Start recording failed: {}", err)));
            return self.result;
        }

        // Execute second function call
        let second_call = match self.executor.to_mut().transact_raw(
            config.from_for_fn2,
            config.addr_contract_with_fn2,
            config.calldata_fn2,
            U256::ZERO,
        ) {
            Ok(result) => result,
            Err(err) => {
                self.result.single_fail(Some(format!("Second function call failed: {}", err)));
                return self.result;
            }
        };

        // Stops recording the EVM state changes with 'vm.stopAndReturnStateDiff()'
        let stop_recording_calldata = Bytes::from_static(&[0xaa, 0x5c, 0xf9, 0x0e]);
        let second_diff_result = match self.executor.to_mut().transact_raw(self.tcfg.sender, CHEATCODE_ADDRESS, stop_recording_calldata, U256::ZERO) {
            Ok(result) => result,
            Err(err) => {
                self.result.single_fail(Some(format!("Stop recording failed: {}", err)));
                return self.result;
            }
        };

        // Decode the state diff result
        let second_diff: Vec<AccountAccess> = match alloy_sol_types::SolValue::abi_decode(&second_diff_result.result, false) {
            Ok(diff) => diff,
            Err(e) => {
                self.result.single_fail(Some(format!("Failed to decode second state diff: {}", e)));
                return self.result;
            }
        };

        // eprintln!("second_state_diffs: {:?}", second_diff);

        let second_state_changes = process_state_diffs(&second_diff, config.addr_contract_with_fn2);

        // Compare all state changes
        let comparison_result = compare_all_state_changes(&first_state_changes, &second_state_changes);

        
        let success_compare_own_state_differences = if config.compare_own_state_differences {
            comparison_result.own_changes.balance_diff.is_none()
            && comparison_result.own_changes.storage_diffs.is_empty()
        } else {
            true
        };

        let success_compare_returned_values = if config.compare_returned_values {
            first_call.result == second_call.result
        } else {
            true
        };

        let success = success_compare_own_state_differences
            && success_compare_returned_values
            && comparison_result.external_changes.is_empty()
            && !first_call.reverted 
            && !second_call.reverted;

        if !success {
            eprintln!("first_call.reverted={}", first_call.reverted);
            eprintln!("second_call.reverted={}", second_call.reverted);

            eprintln!("first_state_changes");
            eprintln!("first_own_storage_changes: {:?}", first_state_changes.own_storage_changes);
            eprintln!("first_external_storage_changes: {:?}", first_state_changes.external_storage_changes);

            eprintln!("second_state_changes");
            eprintln!("second_own_storage_changes: {:?}", second_state_changes.own_storage_changes);
            eprintln!("second_external_storage_changes: {:?}", second_state_changes.external_storage_changes);

            eprintln!("State differences:");
            eprintln!("\nOwn contract changes:");
            if let Some(balance_diff) = &comparison_result.own_changes.balance_diff {
                eprintln!("Balance changes:");
                eprintln!("  First:  {:?} -> {:?}", balance_diff.first_old, balance_diff.first_new);
                eprintln!("  Second: {:?} -> {:?}", balance_diff.second_old, balance_diff.second_new);
            }
            for (slot, storage_diff) in &comparison_result.own_changes.storage_diffs {
                eprintln!("Storage slot {:?}:", slot);
                eprintln!("  First:  {:?} -> {:?}", storage_diff.first_old, storage_diff.first_new);
                eprintln!("  Second: {:?} -> {:?}", storage_diff.second_old, storage_diff.second_new);
            }

            eprintln!("\nExternal contracts changes:");
            for (addr, changes) in &comparison_result.external_changes {
                eprintln!("\nAddress: {:?}", addr);
                if let Some(balance_diff) = &changes.balance_diff {
                    eprintln!("Balance changes:");
                    eprintln!("  First:  {:?} -> {:?}", balance_diff.first_old, balance_diff.first_new);
                    eprintln!("  Second: {:?} -> {:?}", balance_diff.second_old, balance_diff.second_new);
                }
                for (slot, storage_diff) in &changes.storage_diffs {
                    eprintln!("Storage slot {:?}:", slot);
                    eprintln!("  First:  {:?} -> {:?}", storage_diff.first_old, storage_diff.first_new);
                    eprintln!("  Second: {:?} -> {:?}", storage_diff.second_old, storage_diff.second_new);
                }
            }
        }

        self.result.single_result(success, reason, get_config_raw_call_result);
        self.result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValueChange {
    old: alloy_primitives::FixedBytes<32>,
    new: alloy_primitives::FixedBytes<32>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct StorageSlot(alloy_primitives::FixedBytes<32>);

#[derive(Debug)]
struct StateChangesByAddress {
    balance: ValueChange,
    storage: std::collections::HashMap<StorageSlot, ValueChange>,
}

#[derive(Debug)]
struct StateChanges {
    own_storage_changes: StateChangesByAddress,
    external_storage_changes: std::collections::HashMap<Address, StateChangesByAddress>,
}

fn process_state_diffs(diff: &[AccountAccess], contract_address: Address) -> StateChanges {
    let mut own_storage_changes: Option<StateChangesByAddress> = None;
    let mut external_storage_changes: std::collections::HashMap<Address, StateChangesByAddress> = std::collections::HashMap::new();

    for access in diff.iter().filter(|a| !a.reverted) {
        // let mut storage_changes: std::collections::HashMap<StorageSlot, ValueChange> = std::collections::HashMap::new();
        
        let old_balance = access.oldBalance;
        let new_balance = access.newBalance;

        if access.account == contract_address {
            match &mut own_storage_changes {
                Some(state_change) => {
                    state_change.balance = ValueChange {
                        old: state_change.balance.old,
                        new: alloy_primitives::FixedBytes::from(new_balance.to_be_bytes()),
                    };
                }
                None => {
                    own_storage_changes = Some(StateChangesByAddress {
                        balance: ValueChange { 
                            old: alloy_primitives::FixedBytes::from(old_balance.to_be_bytes()), 
                            new: alloy_primitives::FixedBytes::from(new_balance.to_be_bytes()) 
                        },
                        storage: std::collections::HashMap::new(),
                    });
                }
            }
        }
        else {
            if !external_storage_changes.contains_key(&access.account) {
                external_storage_changes.insert(access.account, StateChangesByAddress {
                    balance: ValueChange { 
                        old: alloy_primitives::FixedBytes::from(old_balance.to_be_bytes()), 
                        new: alloy_primitives::FixedBytes::from(new_balance.to_be_bytes()) 
                    },
                    storage: std::collections::HashMap::new(),
                });
            }
            external_storage_changes.get_mut(&access.account).unwrap().balance = ValueChange {
                old: external_storage_changes.get(&access.account).unwrap().balance.old,
                new: alloy_primitives::FixedBytes::from(new_balance.to_be_bytes()),
            };
        }        

        // Process storage changes
        for s in access.storageAccesses.iter().filter(|s| s.isWrite) {
            // eprintln!("s: {:?}", s);
            let slot = StorageSlot(s.slot);
            let account = s.account;

            let old_value = U256::from_be_bytes(s.previousValue.0);
            let new_value = U256::from_be_bytes(s.newValue.0);
            
            let storage_changes;    
            if access.account == contract_address {
                storage_changes = &mut own_storage_changes.as_mut().unwrap().storage;
            } else {
                storage_changes = &mut external_storage_changes.get_mut(&account).unwrap().storage;
            }

            if !storage_changes.contains_key(&slot) {
                storage_changes.insert(slot, ValueChange {
                    old: alloy_primitives::FixedBytes::from(old_value.to_be_bytes()),
                    new: alloy_primitives::FixedBytes::from(new_value.to_be_bytes()),
                });
            } else {
                storage_changes.insert(slot, ValueChange {
                    old: storage_changes.get(&slot).unwrap().old,
                    new: alloy_primitives::FixedBytes::from(new_value.to_be_bytes()),
                });
            }
        }
    }

    // А что если не будет и вовсе own_storage_changes???
    StateChanges {
        own_storage_changes: own_storage_changes.unwrap(),
        external_storage_changes: external_storage_changes,
    }
}

#[derive(Debug)]
struct BalanceComparison {
    first_old: alloy_primitives::FixedBytes<32>,
    first_new: alloy_primitives::FixedBytes<32>,
    second_old: alloy_primitives::FixedBytes<32>,
    second_new: alloy_primitives::FixedBytes<32>,
}

#[derive(Debug)]
struct StorageComparison {
    first_old: alloy_primitives::FixedBytes<32>,
    first_new: alloy_primitives::FixedBytes<32>,
    second_old: alloy_primitives::FixedBytes<32>,
    second_new: alloy_primitives::FixedBytes<32>,
}

#[derive(Debug)]
struct StateDiffComparison {
    balance_diff: Option<BalanceComparison>,
    storage_diffs: std::collections::HashMap<StorageSlot, StorageComparison>
}

fn compare_state_changes(
    first: &StateChangesByAddress,
    second: &StateChangesByAddress,
) -> StateDiffComparison {
    let mut result = StateDiffComparison {
        balance_diff: None,
        storage_diffs: std::collections::HashMap::new(),
    };

    // Compare balances
    if first.balance.old != second.balance.old || first.balance.new != second.balance.new {
        result.balance_diff = Some(BalanceComparison {
            first_old: first.balance.old,
            first_new: first.balance.new,
            second_old: second.balance.old,
            second_new: second.balance.new,
        });
    }

    // Compare storage values
    let all_slots: std::collections::HashSet<_> = first.storage.keys()
        .chain(second.storage.keys())
        .collect();

    for slot in all_slots {
        let first_values = first.storage.get(slot)
            .map(|v| (v.old, v.new))
            .unwrap_or_default();
        let second_values = second.storage.get(slot)
            .map(|v| (v.old, v.new))
            .unwrap_or_default();
        
        if first_values != second_values {
            result.storage_diffs.insert(*slot, StorageComparison {
                first_old: first_values.0,
                first_new: first_values.1,
                second_old: second_values.0,
                second_new: second_values.1,
            });
        }
    }

    result
}

#[derive(Debug)]
struct ComparisonResult {
    own_changes: StateDiffComparison,
    external_changes: std::collections::HashMap<Address, StateDiffComparison>,
}

fn compare_all_state_changes(
    first_changes: &StateChanges,
    second_changes: &StateChanges,
) -> ComparisonResult {
    let mut result = ComparisonResult {
        own_changes: compare_state_changes(&first_changes.own_storage_changes, &second_changes.own_storage_changes),
        external_changes: std::collections::HashMap::new(),
    };

    // Collect all unique addresses from both states
    let all_addresses: std::collections::HashSet<_> = first_changes.external_storage_changes.keys()
        .chain(second_changes.external_storage_changes.keys())
        .collect();

    // Compare changes for each address
    for addr in all_addresses {
        //TODO: исправить
        let empty_state = StateChangesByAddress {
            balance: ValueChange { old: alloy_primitives::FixedBytes::ZERO, new: alloy_primitives::FixedBytes::ZERO },
            storage: std::collections::HashMap::new(),
        };
        
        let first_state = first_changes.external_storage_changes.get(addr)
            .unwrap_or(&empty_state);
        
        let second_state = second_changes.external_storage_changes.get(addr)
            .unwrap_or(&empty_state);

        let comparison = compare_state_changes(first_state, second_state);
        
        // Only include addresses that have differences
        if comparison.balance_diff.is_some() || !comparison.storage_diffs.is_empty() {
            result.external_changes.insert(*addr, comparison);
        }
    }

    result
}

fn fuzzer_with_cases(
    seed: Option<U256>,
    cases: u32,
    max_global_rejects: u32,
    file_failure_persistence: Option<Box<dyn FailurePersistence>>,
) -> TestRunner {
    let config = proptest::test_runner::Config {
        failure_persistence: file_failure_persistence,
        cases,
        max_global_rejects,
        // Disable proptest shrink: for fuzz tests we provide single counterexample,
        // for invariant tests we shrink outside proptest.
        max_shrink_iters: 0,
        ..Default::default()
    };

    if let Some(seed) = seed {
        trace!(target: "forge::test", %seed, "building deterministic fuzzer");
        let rng = TestRng::from_seed(RngAlgorithm::ChaCha, &seed.to_be_bytes::<32>());
        TestRunner::new_with_rng(config, rng)
    } else {
        trace!(target: "forge::test", "building stochastic fuzzer");
        TestRunner::new(config)
    }
}

/// Holds data about a persisted invariant failure.
#[derive(Serialize, Deserialize)]
struct InvariantPersistedFailure {
    /// Recorded counterexample.
    call_sequence: Vec<BaseCounterExample>,
    /// Bytecode of the test contract that generated the counterexample.
    #[serde(skip_serializing_if = "Option::is_none")]
    driver_bytecode: Option<Bytes>,
}

/// Helper function to load failed call sequence from file.
/// Ignores failure if generated with different test contract than the current one.
fn persisted_call_sequence(path: &Path, bytecode: &Bytes) -> Option<Vec<BaseCounterExample>> {
    foundry_common::fs::read_json_file::<InvariantPersistedFailure>(path).ok().and_then(
        |persisted_failure| {
            if let Some(persisted_bytecode) = &persisted_failure.driver_bytecode {
                // Ignore persisted sequence if test bytecode doesn't match.
                if !bytecode.eq(persisted_bytecode) {
                    let _= sh_warn!("\
                            Failure from {:?} file was ignored because test contract bytecode has changed.",
                        path
                    );
                    return None;
                }
            };
            Some(persisted_failure.call_sequence)
        },
    )
}

/// Helper functions to return canonicalized invariant failure paths.
fn invariant_failure_paths(
    config: &InvariantConfig,
    contract_name: &str,
    invariant_name: &str,
) -> (PathBuf, PathBuf) {
    let dir = config
        .failure_persist_dir
        .clone()
        .unwrap()
        .join("failures")
        .join(contract_name.split(':').next_back().unwrap());
    let dir = canonicalized(dir);
    let file = canonicalized(dir.join(invariant_name));
    (dir, file)
}