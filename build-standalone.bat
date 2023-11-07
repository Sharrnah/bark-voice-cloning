pip3 install -U pyinstaller

rem install plugin dependencies (also added in pyinstaller)
rem pip3 install winsdk

pyinstaller barkVoiceClone.py -y ^
            --python-option=u ^
            --hidden-import=pytorch --collect-data torch --copy-metadata torch ^
            --hidden-import=torchaudio.lib.libtorchaudio ^
            --hidden-import=hydra ^
            --copy-metadata numpy ^
            --copy-metadata tqdm ^
            --copy-metadata regex ^
            --copy-metadata requests ^
            --copy-metadata packaging ^
            --copy-metadata filelock ^
            --copy-metadata huggingface-hub ^
            --copy-metadata safetensors ^
            --copy-metadata pyyaml ^
            --copy-metadata transformers ^
            --collect-all torchaudio ^
            --collect-all fairseq ^
            --collect-all transformers ^
            --collect-submodules hydra ^
            --collect-data hydra ^
            --collect-submodules hydra._internal.core_plugins ^
            --collect-submodules fairseq ^
            --onedir --clean --distpath dist
rem --copy-metadata rich
rem --collect-all lazy_loader
rem --collect-all decorator
rem --collect-all librosa
rem --collect-all torchlibrosa

rem pyinstaller barkVoiceClone.spec -y
