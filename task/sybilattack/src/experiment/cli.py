from lightning.pytorch.cli import LightningCLI


class SybilCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(("data.source_transform", "data.transform_type"),
                              "model.predict_prefix",
                              compute_fn=lambda source_transform, transform_type: f"{source_transform}_{transform_type}",)
        parser.link_arguments(("data.source_transform", "data.target_transform"),
                              "model.same",
                              compute_fn=lambda source_transform, target_transform: source_transform == target_transform,)
        parser.link_arguments("data.transform_type",
                              "model.transform_type")
