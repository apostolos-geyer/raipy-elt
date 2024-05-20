import logging
from collections import defaultdict
from typing import (
    Any,
    Callable,
    NamedTuple,
    ParamSpec,
    TypeAlias,
    TypeVar,
    overload,
)

from attrs import define, field

P = ParamSpec("P")
R = TypeVar("R")
StageKey: TypeAlias = str
StageResults: TypeAlias = dict[str, R]
ResultMapper: TypeAlias = Callable[[R], StageResults]


def DEFAULT_RESULT_MAPPER(x: R) -> StageResults:
    return {"return": x}


class ResultMapping(NamedTuple):
    """
    Defines a mapping of an output from a stage to an argument in another stage
    If `from_result` is unset, the assumption will be to look under the key 'return' of the stage,
    which is the default key for the return value of the function.
    """

    from_stage: StageKey
    as_arg: str
    from_result: str = "return"


class ParamMapping(NamedTuple):
    """
    Defines a mapping of a parameter from the pipeline to an argument in a stage
    """

    from_param: str
    as_arg: str


@define
class Stage:
    func: Callable[P, R]
    has_run: bool = field(default=False)
    result_inputs: list["ResultMapping"] = field(factory=list)
    param_inputs: list["ParamMapping"] = field(factory=list)
    defaults: dict[str, Any] = field(factory=dict)
    result_mapper: ResultMapper = field(default=DEFAULT_RESULT_MAPPER)
    results: StageResults = field(factory=dict)

    def reset(self) -> None:
        """
        resets all internal state except for defaults
        """
        self.has_run = False
        self.results = {}


class ExitPipeline(Exception):
    """
    Exception to exit the pipeline
    """

    def __init__(self, message: str, error: False, *args):
        self.message = message
        super().__init__(*args)


@define
class Pipeline:
    UNSET = "_UNSET"

    name: str = field(default="Pipeline")
    parameter_set: set[str] = field(factory=set)
    parameters: defaultdict[str, Any] = field(
        factory=lambda: defaultdict(lambda: Pipeline.UNSET)
    )
    variables: defaultdict[str, Any] = field(
        factory=lambda: defaultdict(lambda: Pipeline.UNSET)
    )
    stages: dict[StageKey, Stage] = field(default={})
    logger: logging.Logger = field(init=False)

    @logger.default
    def _logger_default(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(self.name)

    def reset(self) -> None:
        """
        resets all internal state except for parameters and variables
        """

        for stage_key in self.stages:
            stage = self.stages[stage_key]
            stage.reset()
            self.stages[stage_key] = stage

        self.logger = logging.getLogger(self.name)
        self.logger.info("Pipeline has been reset")

    @classmethod
    def define(
        cls,
        name: str,
        parameters: set[str] | None = None,
        **parameter_defaults,
    ):
        """
        Define a new pipeline with the given name and parameters

        :param name: the name of the pipeline
        :param parameters: a dictionary of parameter names and types
        :param parameter_defaults: default values for the parameters
        :returns: the pipeline object which can be used to define stages
        """

        parameter_set = {
            name
            for name in set(parameters or []).union(
                set(parameter_defaults.keys() or [])
            )
        }
        starting_parameters = defaultdict(lambda: cls.UNSET, parameter_defaults)

        return Pipeline(
            name=name, parameters=starting_parameters, parameter_set=parameter_set
        )

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Sets a parameter in the pipeline

        :param name: the name of the parameter
        :param value: the value of the parameter
        """
        if name not in self.parameter_set:
            self.logger.warn(
                f"Parameter {name} not found in pipeline parameters, ignoring"
            )
            return

        self.parameters[name] = value

    def set_parameters(self, **kwargs: Any) -> None:
        """
        Sets multiple parameters in the pipeline

        :param kwargs: a dictionary of parameters and values
        """
        self.parameters.update(kwargs)

    def get_parameter(self, name: str, must=False) -> Any:
        """
        Gets a parameter from the pipeline

        :param name: the name of the parameter
        :param must: whether to raise an error if the parameter is not found
        :returns: the value of the parameter
        """
        if not (value := self.parameters.get(name)) and must:
            raise ValueError(f"Parameter {name} not found in pipeline parameters")
        elif value is None:
            self.logger.warn(f"Parameter {name} not found in pipeline parameters")
            return self.UNSET
        return value

    def get_stage_result(
        self, from_stage: StageKey, from_result: str = "return", must=False
    ) -> StageResults:
        """
        Gets the result of a stage under the given key

        :param stage_key: the key of the stage
        :param result_key: the key of the result
        :param must: whether to raise an error if the result is not found
        :returns: the results of the stage
        """
        if ((stage := self.stages.get(from_stage)) is None) and must:
            raise ValueError(f"No stage with key {from_stage} found in the pipeline")
        elif not stage:
            self.logger.warn(
                f"Attempting to get result {from_result} from stage {from_stage} but no stage with key {from_stage} found in the pipeline"
            )
            return self.UNSET
        elif not (has_run := stage.has_run) and must:
            raise ValueError(f"Stage {from_stage} has not run yet")
        elif not has_run:
            self.logger.warn(
                f"Attempting to get result {from_result} from stage {from_stage} but stage {from_stage} has not run yet"
            )
            return self.UNSET

        # stage exists and has run

        if ((result := stage.results.get(from_result)) is None) and must:
            raise ValueError(f"Result {from_result} not found in stage {from_stage}")
        elif result is None:
            return self.UNSET

        return result

    def get_stage_results(
        self, stage_key: StageKey, result_keys: str = "all", must=False
    ) -> StageResults:
        """
        Gets the results of a stage under the given key

        :param stage_key: the key of the stage
        :param result_keys: the keys of the results to get, if 'all', gets all results
        :param must: whether to raise an error if the result is not found
        :returns: the results of the stage
        """

        if not (stage := self.stages.get(stage_key)) and must:
            raise ValueError(f"No stage with key {stage_key} found in the pipeline")
        elif not stage:
            self.logger.warn(
                f"Attempting to get results {result_keys} from stage {stage_key} but no stage with key {stage_key} found in the pipeline"
            )
            return {"return": self.UNSET}
        elif not (has_run := stage.has_run) and must:
            raise ValueError(f"Stage {stage_key} has not run yet")
        elif not has_run:
            self.logger.warn(
                f"Attempting to get results {result_keys} from stage {stage_key} but stage {stage_key} has not run yet"
            )
            return {"return": self.UNSET}

        # stage exists and has run

        if result_keys == "all":
            return stage.results
        else:
            return {
                key: self.get_stage_result(stage_key, key, must) for key in result_keys
            }

    @overload
    def stage(
        self,
        name: str | None = None,
        *,
        use_params: list[ParamMapping] | None = None,
        use_outputs: list[ResultMapping] | None = None,
        result_mapper: ResultMapper = DEFAULT_RESULT_MAPPER,
        **defaults: P.kwargs,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Registers a stage in the pipeline under the stage key `name` or under the name of the function if `name` is not provided

        :param name: the name of the stage
        :param use_params: the parameters to use from the pipeline,
            a list of tuples of the form (from_param: the name of the parameter, as_arg: the argument to pass the parameter as)
        :param use_outputs: the outputs to use from the pipeline,
            a list of tuples of the form (from_stage: the key of the stage, from_output: the output to use, as_arg: the argument to pass the output as)
        :param defaults: default values for arguments to the stage

        :returns: a decorator that registers the function as a stage in the pipeline
        """
        ...

    @overload
    def stage(
        self,
        func: Callable[P, R],
    ):
        """
        Registers a stage in the pipeline under the name of the function

        :param func: the function to register as a stage

        :returns: the function
        """
        ...

    def stage(
        self,
        name_or_callable: str | Callable[P, R] | None = None,
        *,
        use_params: list[ParamMapping] | None = None,
        use_outputs: list[ResultMapping] | None = None,
        result_mapper: ResultMapper = None,
        **defaults: P.kwargs,
    ) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
        """
        Registers a stage in the pipeline under the stage key `name` or under the name of the function if `name` is not provided

        :param name_or_callable: the name of the stage or the function to register as a stage
        :param use_params: the parameters to use from the pipeline,
            a list of tuples of the form (from_param: the name of the parameter, as_arg: the argument to pass the parameter as)
        :param use_outputs: the outputs to use from the pipeline,
            a list of tuples of the form (from_stage: the key of the stage, from_output: the output to use, as_arg: the argument to pass the output as)
        :param result_mapper: a function to map the result of the stage to a dictionary of results
        :param defaults: default values for arguments to the stage


        :returns: a decorator that registers the function as a stage in the pipeline if `name_or_callable` is a function, otherwise returns the function itself

        """

        stage_key = None

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            nonlocal stage_key
            if stage_key is None:
                stage_key
                stage_key = func.__name__

            match (use_params, use_outputs, result_mapper):
                case (None, None, None):
                    self.stages[stage_key] = Stage(func=func, defaults=defaults)
                case (None, None, _):
                    self.stages[stage_key] = Stage(
                        func=func, defaults=defaults, result_mapper=result_mapper
                    )
                case (None, _, None):
                    self.stages[stage_key] = Stage(
                        func=func, defaults=defaults, result_inputs=use_outputs
                    )
                case (None, _, _):
                    self.stages[stage_key] = Stage(
                        func=func,
                        defaults=defaults,
                        result_inputs=use_outputs,
                        result_mapper=result_mapper,
                    )
                case (_, None, None):
                    self.stages[stage_key] = Stage(
                        func=func, defaults=defaults, param_inputs=use_params
                    )
                case (_, None, _):
                    self.stages[stage_key] = Stage(
                        func=func,
                        defaults=defaults,
                        param_inputs=use_params,
                        result_mapper=result_mapper,
                    )
                case (_, _, None):
                    self.stages[stage_key] = Stage(
                        func=func,
                        defaults=defaults,
                        param_inputs=use_params,
                        result_inputs=use_outputs,
                    )
                case (_, _, _):
                    self.stages[stage_key] = Stage(
                        func=func,
                        defaults=defaults,
                        param_inputs=use_params,
                        result_inputs=use_outputs,
                        result_mapper=result_mapper,
                    )

            return func

        if isinstance(name_or_callable, str):
            stage_key = name_or_callable
            return decorator
        elif name_or_callable is None:
            return decorator
        elif callable(name_or_callable):
            stage_key = name_or_callable.__name__
            return decorator(name_or_callable)

    def run_stage(self, stage_key: StageKey, **overrides) -> None:
        """
        Runs the stage with the given key. Uses the parameters and outputs defined in the stage to run it.
        If the stage has already run, it will not run again.
        If it depends on outputs from a stage that has not run yet, it will not run.

        :param stage_key: the key of the stage to run
        """
        stage = self.stages.get(stage_key)
        if not stage:
            raise ValueError(f"No stage with key {stage_key} found in the pipeline")

        kwds = defaultdict(lambda: self.UNSET)
        kwds.update(stage.defaults)

        self.logger.info(
            f"Running stage {stage_key} with default arguments {stage.defaults}"
        )

        if param_mappings := stage.param_inputs:
            for param_mapping in param_mappings:
                from_param = param_mapping.from_param
                as_arg = param_mapping.as_arg

                if (param := self.get_parameter(from_param)) is self.UNSET:
                    self.logger.warn(
                        f"Parameter {from_param} for argument {as_arg} not found in pipeline parameters, ignoring"
                    )
                    continue

                if (value := stage.defaults.get(as_arg)) is not None:
                    self.logger.debug(
                        f"Parameter {from_param} for {as_arg} overriding a default value {value}"
                    )

                kwds[as_arg] = param

        if result_mappings := stage.result_inputs:
            for result_mapping in result_mappings:
                from_stage = result_mapping.from_stage
                from_result = result_mapping.from_result
                as_arg = result_mapping.as_arg

                if (
                    result := self.get_stage_result(from_stage, from_result, must=False)
                ) is self.UNSET:
                    self.logger.warn(
                        f"Output {from_result} in stage {from_stage} for argument {as_arg} not found, ignoring"
                    )
                    continue

                if (value := stage.defaults.get(as_arg)) is not None:
                    self.logger.debug(
                        f"Output {from_stage} for {as_arg} overriding a default value {value}"
                    )

                kwds[as_arg] = result

        try:
            self.logger.info(f"Calling {stage_key} with finalized arguments {kwds}")
            result = stage.func(**kwds)
            self.logger.info(f"Stage {stage_key} has run successfully")
            stage.has_run = True
            stage.results = stage.result_mapper(result)
            self.logger.info(f"Stage {stage_key} results: {stage.results}")
        except ExitPipeline as ep:
            raise ep
        except Exception as e:
            self.logger.error(
                f"An error occurred while running stage {stage_key}: {e}", exc_info=True
            )
            return

    def run(self):
        """
        Runs the pipeline
        """
        for stage_key in self.stages:
            try:
                self.logger.info(f"Running stage {stage_key}")
                self.run_stage(stage_key)
            except ExitPipeline as ep:
                self.logger.info(f"Pipeline has exited: {ep.message}")
                return

        self.logger.info("Pipeline has run successfully")
