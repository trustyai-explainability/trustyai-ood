"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20230824
© Copyright IBM Corp. 2024 All Rights Reserved.
"""
import json
from marshmallow import Schema, fields, validates_schema, ValidationError


class IBMCosSchema(Schema):
    api_key = fields.Str(required=True, description="COS api key")
    resource_instance_id = fields.Str(required=True, description="COS resource instance ID")
    service_endpoint = fields.Str(required=True, description="Endpoint to reach COS instance")
    auth_endpoint = fields.Str(required=True, description="COS authentication endpoint")
    bucket = fields.Str(required=True, description="Bucket to access desired asset")
    file_path = fields.Str(required=True, description="file path to desired artifact in bucket")

    class Meta:
        ordered = True


class ModelMetadataSchema(Schema):
    type = fields.Str(required=True, description="Model's ML framework (tf | pytorch)")
    arch = fields.Str(required=True, description="Model's NN architecture (i.e. resnet50)")

    class Meta:
        ordered = True


class LocationSchema(Schema):
    uri = fields.Str(required=False, description="URI to asset; should be available publicly available or pre-signed")
    ibm_cos = fields.Nested(IBMCosSchema, required=False)

    @validates_schema
    def validate_b_requires_a(self, data):
        if 'uri' not in data and 'ibm_cos' not in data:
            raise ValidationError('uri OR ibm_cos is required')


class ModelRefSchema(Schema):
    metadata = fields.Nested(ModelMetadataSchema, required=True)
    location = fields.Nested(LocationSchema, required=True)

    class Meta:
        ordered = True


class DataRefSchema(Schema):
    # type = fields.Str(required=True, description="Type of dataset , i.e. image, audio.")
    metadata = fields.Dict(required=True, keys=fields.Str, description="Info about dataset for preprocessing specific "
                                                                       "to dataset type")
    location = fields.Nested(LocationSchema, required=True)

    class Meta:
        ordered = True


class OutputSchema(Schema):
    location = fields.Nested(LocationSchema, required=True)
    save_format = fields.Str(required=False,
                             allowableValues={"native", "onnx"},
                             description="Output save format. should be native or onnx. Will use 'native' if unknown "
                                         "value given")


class OODEnableSchema(Schema):
    model_ref = fields.Nested(ModelRefSchema, required=True)
    data_ref = fields.Nested(DataRefSchema, required=False)
    output_ref = fields.Nested(OutputSchema, required=True)

    class Meta:
        ordered = True


def generate_api_spec(path, app, spec, views):
    spec.components.schema("OODEnableSchema", schema=OODEnableSchema)

    with app.test_request_context():
        for view in views:
            spec.path(view=view)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(spec.to_dict(), f, ensure_ascii=False, indent=4)
