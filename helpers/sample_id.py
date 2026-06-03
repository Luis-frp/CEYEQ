def build_sample_id(filepath, class_name=None):
    del class_name
    return str(filepath).replace("\\", "/").strip()
