def load_config(dataset_key='pilot', verbose=False):

    import toml
    from pathlib import Path

    package_dir = Path(__file__).parent.absolute()
    config_file = package_dir / "config.toml"
    print(f'Paths are for dataset tagged: {dataset_key}')
    if verbose:
        print(f'Package dir: {package_dir}')
        print(f'config file: {config_file}')

    config = dict()
    toml_dict = toml.load(config_file)
    config.update(toml_dict[dataset_key])
    config.update({'package_dir':package_dir,'config_file':config_file})
    not_found = []
    for key in config:
        if Path(config[key]).exists():
            config[key] = Path(config[key])

    for key in not_found:
        print(f'Did not find {key}')
        config.pop(key, None)

    return config
