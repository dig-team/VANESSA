conda create -y --prefix ./environments/rst_seg python=3.7
conda run --prefix ./environments/rst_seg pip install -r ./segbot/requirements.txt
conda run --prefix ./environments/rst_seg python -c "import nltk; nltk.download('punkt')"

conda create -y --prefix ./environments/rst_parsing python=3.8
conda run --prefix ./environments/rst_parsing pip install -r ./parsing/requirements.txt
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14CPKfvaGolg5Kd0smBpb2B4un3ZBaJgQ" -O parsing/best.ckpt && rm -rf /tmp/cookies.txt