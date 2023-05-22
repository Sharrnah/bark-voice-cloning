pip3 install -U pyinstaller

rem install plugin dependencies (also added in pyinstaller)
rem pip3 install winsdk

pyinstaller barkVoiceClone.py -y ^
            --python-option=u ^
            --hidden-import=pytorch --collect-data torch --copy-metadata torch ^
            --hidden-import=torchaudio.lib.libtorchaudio ^
            --hidden-import=hydra ^
            --copy-metadata numpy ^
            --collect-all torchaudio ^
            --collect-all fairseq ^
            --collect-submodules hydra ^
            --collect-data hydra ^
            --collect-submodules hydra._internal.core_plugins ^
            --collect-submodules fairseq

rem pyinstaller barkVoiceClone.spec -y
