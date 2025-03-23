from omegaconf import OmegaConf


def get_config(config_class):
    cfg = OmegaConf.structured(config_class)

    additional_cfg = OmegaConf.from_cli()
    if "yaml" in additional_cfg:
        yaml_cfg = OmegaConf.load(additional_cfg.yaml)
        yaml_cfg = OmegaConf.masked_copy(yaml_cfg, cfg.keys())
        additional_cfg = OmegaConf.merge(yaml_cfg, additional_cfg)
        additional_cfg.pop("yaml")
    # additional_cfg = OmegaConf.masked_copy(additional_cfg, cfg.keys())

    cfg = OmegaConf.to_object(OmegaConf.merge(cfg, additional_cfg))
    return cfg
