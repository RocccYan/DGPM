# -*- coding: utf-8 -*-


from abc import ABC
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict

import dataclasses
import json
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


class HfArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field_name = f"--{field.name}"
            kwargs = field.metadata.copy()
            # field.metadata is not used at all by Data Classes,
            # it is provided as a third-party extension mechanism.
            if isinstance(field.type, str):
                raise ImportError(
                    "This implementation is not compatible with Postponed Evaluation of Annotations (PEP 563),"
                    "which can be opted in from Python 3.7 with `from __future__ import annotations`."
                    "We will add compatibility when Python 3.9 is released."
                )
            typestring = str(field.type)
            for prim_type in (int, float, str):
                for collection in (List,):
                    if (
                        typestring == f"typing.Union[{collection[prim_type]}, NoneType]"
                        or typestring == f"typing.Optional[{collection[prim_type]}]"
                    ):
                        field.type = collection[prim_type]
                if (
                    typestring == f"typing.Union[{prim_type.__name__}, NoneType]"
                    or typestring == f"typing.Optional[{prim_type.__name__}]"
                ):
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
            elif field.type is bool or field.type is Optional[bool]:
                if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                    kwargs["action"] = "store_false" if field.default is True else "store_true"
                if field.default is True:
                    field_name = f"--no_{field.name}"
                    kwargs["dest"] = field.name
            elif hasattr(field.type, "__origin__") and issubclass(field.type.__origin__, List):
                kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]
                assert all(
                    x == kwargs["type"] for x in field.type.__args__
                ), "{} cannot be a List of mixed types".format(field.name)
                if field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
            else:
                kwargs["type"] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    kwargs["default"] = field.default_factory()
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, look_for_args_file=True, args_filename=None
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            return (*outputs,)

    def parse_json_file(self, json_file: str) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def parse_dict(self, args: dict) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.
        """
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in args.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)


"""### Define Arguments"""

@dataclass
class DataArguments:
    dataset_name: str = field(default='dataset',
                      metadata={"help": 'Name the Data.'})
    feature_file: str = field(default="./data/pheme-2/feature.json",
                            metadata={"help": "The path of train file"})
    label_file: str = field(default="./data/pheme-2/label.json",
                            metadata={"help": "The path of train file"})
    dataset_root: str = field(default='./data/',
                              metadata={"help": "The root path of processed dataset."})
    num_classes: int = field(default=2,
                             metadata={"help":"The number of classes of the data."})


@dataclass
class ModelArguments:
    model_name: str = field(default='Model',
                            metadata={"help": 'Name the Model.'})
    model_name_node: str = field(default="PROTEINS_graphmae")
    model_name_motif: str = field(default="PROTEINS_D=512")
    model_path: str = field(default="./models/bert-base-uncased",
                            metadata={"help": "The name or path of pre-trained model"})
    eta: float = field(default=1.0,
                       metadata={"help":"the perturb factor of encoders' params."})
    input_channels: int = field(default=6,
                                metadata={"help":"input channels, which follows the dim of nodes features."})
    hidden_channels: int = field(default=128,
                            metadata={"help":"hidden channels."})
    num_layers: int = field(default=2,
                            metadata={"help":"number of gnn layers."})
    readout: str = field(default='global_mean',
                         metadata={"help": "readout from graph, option: all, global_mean, global_max"})


@dataclass
class TrainArguments:
    task_name: str = field(default='task',
                           metadata={"help": "Task name."})
    train_mode: str = field(default='cat')
    run_name: str = field(default="default_run")
    seed: int = field(default=42,
                      metadata={"help": "set random state."})
    device: str = field(default='gpu',
                      metadata={"help": "set random state."})
    cl_mode: str = field(default='simgrace', metadata={"help": "assign contrastive learning mode, simgrace by default."})

    num_epochs: int = field(default=200,
                            metadata={"help": "Maximum Number of Epochs to Train the Model."})
    batch_size: int = field(default=128,
                            metadata={"help": "Training batch size."})
    num_workers: int = field(default=1)

    train_proportion: float = field(default=0.8,
                                   metadata={"help": "The proportion of test data."})                   
    test_proportion: float = field(default=0.2,
                                   metadata={"help": "The proportion of test data."})
    temperature: float = field(default=0.05,
                               metadata={"help": "The temperature used to control the weight of contrastive learning loss."})
    lr: float = field(default=0.001,
                                   metadata={"help": "The learning rate."})
    weight_decay: float = field(default=0.0001,
                                   metadata={"help": "The learning rate."})
    drop_pretrain: int = field(default=0,
                                metadata={"help": "whether drop pretrained parameters."})
    print_per_epoch: int = field(default=20,
                                metadata={"help": "Save the model per #epoches."})


