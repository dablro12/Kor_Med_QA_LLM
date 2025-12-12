def load_model(hg_model_id: str):
    if "qwen" in hg_model_id.lower():
        from src.qwen import Qwen
        model = Qwen(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif "gpt" in hg_model_id.lower():
        from src.gpt import GPT
        model = GPT(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif "deepseek" in hg_model_id.lower():
        from src.deepseek import DeepSeek
        model = DeepSeek(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif "gemma" in hg_model_id.lower():
        from src.gemma import Gemma
        model = Gemma(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif "llama" in hg_model_id.lower():
        from src.llama import Llama
        model = Llama(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif "exaone" in hg_model_id.lower():
        from src.exaone import Exaone
        model = Exaone(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif "kanana" in hg_model_id.lower():
        from src.kanana import Kanana
        model = Kanana(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif 'midm' in hg_model_id.lower():
        from src.midm import Midm
        model = Midm(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif 'clovax' in hg_model_id.lower():
        from src.clovax import Clovax
        model = Clovax(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    elif 'upstage' in hg_model_id.lower():
        from src.upstage import Upstage
        model = Upstage(
            hg_model_id=hg_model_id,
            device="cuda",
            cache_dir="/workspace/kor_med_opendataset/hg_cache"
        )
    else:
        raise ValueError(f"Unsupported model: {hg_model_id}")
    
    return model