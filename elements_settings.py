from typing import Dict


ELEMENT_SETTINGS: Dict[str, Dict] = {
    "Temperature": {
        "kelvin_to_celsius": True,
        "normalize": True,
        "k": 6,
        "degree_clamp_min": 1e-6,
    },
    "Cloud": {
        "kelvin_to_celsius": False,
        "normalize": True,
        "k": 6,
        "degree_clamp_min": 1e-6,
    },
    "Humidity": {
        "kelvin_to_celsius": False,
        "normalize": True,
        "k": 3,
        "degree_clamp_min": 1e-6,
    },
    "Wind": {
        "kelvin_to_celsius": False,
        "normalize": True,
        "k": 6,
        "degree_clamp_min": 1e-6,
    },
}

DATASET_TO_ELEMENT = {
    "temperature": "Temperature",
    "cloud_cover": "Cloud",
    "humidity": "Humidity",
    "component_of_wind": "Wind",
}

ELEMENT_TO_DATASET = {v: k for k, v in DATASET_TO_ELEMENT.items()}

ELEMENT_ALIASES = {
    "temperature": "Temperature",
    "temp": "Temperature",
    "cloud": "Cloud",
    "cloud_cover": "Cloud",
    "humidity": "Humidity",
    "wind": "Wind",
    "component_of_wind": "Wind",
}


def normalize_element_name(element: str) -> str:
    key = str(element).strip()
    if key in ELEMENT_SETTINGS:
        return key
    lowered = key.lower()
    if lowered in ELEMENT_ALIASES:
        return ELEMENT_ALIASES[lowered]
    raise ValueError(
        f"Unsupported element '{element}'. Supported: {list(ELEMENT_SETTINGS.keys())}"
    )


def resolve_element_from_config(config: Dict) -> str:
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset_name", "")).strip().lower()
    if dataset_name in DATASET_TO_ELEMENT:
        return DATASET_TO_ELEMENT[dataset_name]

    element = data_cfg.get("element")
    if element:
        return normalize_element_name(element)

    return "Temperature"


def get_element_settings(element: str) -> Dict:
    element_name = normalize_element_name(element)
    return dict(ELEMENT_SETTINGS[element_name])


def get_dataset_name_from_element(element: str) -> str:
    element_name = normalize_element_name(element)
    return ELEMENT_TO_DATASET[element_name]
