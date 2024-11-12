# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: model_service.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC, 5, 27, 2, "", "model_service.proto"
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x13model_service.proto\x12\x0cmodelservice"!\n\x11UploadDataRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t"%\n\x12UploadDataResponse\x12\x0f\n\x07message\x18\x01 \x01(\t"\xc7\x01\n\x11TrainModelRequest\x12\x12\n\nmodel_type\x18\x01 \x01(\t\x12M\n\x0fhyperparameters\x18\x02 \x03(\x0b\x32\x34.modelservice.TrainModelRequest.HyperparametersEntry\x12\x17\n\x0ftarget_variable\x18\x03 \x01(\t\x1a\x36\n\x14HyperparametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01"7\n\x12TrainModelResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t"\x07\n\x05\x45mpty"2\n\x1bListAvailableModelsResponse\x12\x13\n\x0bmodel_types\x18\x01 \x03(\t"0\n\x0ePredictRequest\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t"%\n\x0fPredictResponse\x12\x12\n\nprediction\x18\x01 \x03(\t"&\n\x12\x44\x65leteModelRequest\x12\x10\n\x08model_id\x18\x01 \x01(\t"&\n\x13\x44\x65leteModelResponse\x12\x0f\n\x07message\x18\x01 \x01(\t" \n\x0eStatusResponse\x12\x0e\n\x06status\x18\x01 \x01(\t".\n\x19ListTrainedModelsResponse\x12\x11\n\tmodel_ids\x18\x01 \x03(\t")\n\x15GetPredictionsRequest\x12\x10\n\x08model_id\x18\x01 \x01(\t"-\n\x16GetPredictionsResponse\x12\x13\n\x0bpredictions\x18\x01 \x03(\t"\xc7\x01\n\x12UpdateModelRequest\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12N\n\x0fhyperparameters\x18\x02 \x03(\x0b\x32\x35.modelservice.UpdateModelRequest.HyperparametersEntry\x12\x17\n\x0ftarget_variable\x18\x03 \x01(\t\x1a\x36\n\x14HyperparametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01"&\n\x13UpdateModelResponse\x12\x0f\n\x07message\x18\x01 \x01(\t2\xe4\x05\n\x0cModelService\x12O\n\nUploadData\x12\x1f.modelservice.UploadDataRequest\x1a .modelservice.UploadDataResponse\x12O\n\nTrainModel\x12\x1f.modelservice.TrainModelRequest\x1a .modelservice.TrainModelResponse\x12U\n\x13ListAvailableModels\x12\x13.modelservice.Empty\x1a).modelservice.ListAvailableModelsResponse\x12\x46\n\x07Predict\x12\x1c.modelservice.PredictRequest\x1a\x1d.modelservice.PredictResponse\x12R\n\x0b\x44\x65leteModel\x12 .modelservice.DeleteModelRequest\x1a!.modelservice.DeleteModelResponse\x12;\n\x06Status\x12\x13.modelservice.Empty\x1a\x1c.modelservice.StatusResponse\x12Q\n\x11ListTrainedModels\x12\x13.modelservice.Empty\x1a\'.modelservice.ListTrainedModelsResponse\x12[\n\x0eGetPredictions\x12#.modelservice.GetPredictionsRequest\x1a$.modelservice.GetPredictionsResponse\x12R\n\x0bUpdateModel\x12 .modelservice.UpdateModelRequest\x1a!.modelservice.UpdateModelResponseb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "model_service_pb2", _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    DESCRIPTOR._loaded_options = None
    _globals["_TRAINMODELREQUEST_HYPERPARAMETERSENTRY"]._loaded_options = None
    _globals["_TRAINMODELREQUEST_HYPERPARAMETERSENTRY"]._serialized_options = b"8\001"
    _globals["_UPDATEMODELREQUEST_HYPERPARAMETERSENTRY"]._loaded_options = None
    _globals["_UPDATEMODELREQUEST_HYPERPARAMETERSENTRY"]._serialized_options = b"8\001"
    _globals["_UPLOADDATAREQUEST"]._serialized_start = 37
    _globals["_UPLOADDATAREQUEST"]._serialized_end = 70
    _globals["_UPLOADDATARESPONSE"]._serialized_start = 72
    _globals["_UPLOADDATARESPONSE"]._serialized_end = 109
    _globals["_TRAINMODELREQUEST"]._serialized_start = 112
    _globals["_TRAINMODELREQUEST"]._serialized_end = 311
    _globals["_TRAINMODELREQUEST_HYPERPARAMETERSENTRY"]._serialized_start = 257
    _globals["_TRAINMODELREQUEST_HYPERPARAMETERSENTRY"]._serialized_end = 311
    _globals["_TRAINMODELRESPONSE"]._serialized_start = 313
    _globals["_TRAINMODELRESPONSE"]._serialized_end = 368
    _globals["_EMPTY"]._serialized_start = 370
    _globals["_EMPTY"]._serialized_end = 377
    _globals["_LISTAVAILABLEMODELSRESPONSE"]._serialized_start = 379
    _globals["_LISTAVAILABLEMODELSRESPONSE"]._serialized_end = 429
    _globals["_PREDICTREQUEST"]._serialized_start = 431
    _globals["_PREDICTREQUEST"]._serialized_end = 479
    _globals["_PREDICTRESPONSE"]._serialized_start = 481
    _globals["_PREDICTRESPONSE"]._serialized_end = 518
    _globals["_DELETEMODELREQUEST"]._serialized_start = 520
    _globals["_DELETEMODELREQUEST"]._serialized_end = 558
    _globals["_DELETEMODELRESPONSE"]._serialized_start = 560
    _globals["_DELETEMODELRESPONSE"]._serialized_end = 598
    _globals["_STATUSRESPONSE"]._serialized_start = 600
    _globals["_STATUSRESPONSE"]._serialized_end = 632
    _globals["_LISTTRAINEDMODELSRESPONSE"]._serialized_start = 634
    _globals["_LISTTRAINEDMODELSRESPONSE"]._serialized_end = 680
    _globals["_GETPREDICTIONSREQUEST"]._serialized_start = 682
    _globals["_GETPREDICTIONSREQUEST"]._serialized_end = 723
    _globals["_GETPREDICTIONSRESPONSE"]._serialized_start = 725
    _globals["_GETPREDICTIONSRESPONSE"]._serialized_end = 770
    _globals["_UPDATEMODELREQUEST"]._serialized_start = 773
    _globals["_UPDATEMODELREQUEST"]._serialized_end = 972
    _globals["_UPDATEMODELREQUEST_HYPERPARAMETERSENTRY"]._serialized_start = 257
    _globals["_UPDATEMODELREQUEST_HYPERPARAMETERSENTRY"]._serialized_end = 311
    _globals["_UPDATEMODELRESPONSE"]._serialized_start = 974
    _globals["_UPDATEMODELRESPONSE"]._serialized_end = 1012
    _globals["_MODELSERVICE"]._serialized_start = 1015
    _globals["_MODELSERVICE"]._serialized_end = 1755
# @@protoc_insertion_point(module_scope)