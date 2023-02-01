# Diff-SVC
Singing Voice Conversion via diffusion model


# * 주의사항 *
## 1. **모르는건 물어봐도 되는데 대답이 느릴 수 있습니다**(위쪽에 Issues 에다가 질문있으면 작성해주세요. 유툽댓글은 확인 잘 안합니다.)
## 2. **이 프로젝트는 학술 교류 목적으로 설립되었으며 프로덕션 환경을 위한 것이 아닙니다. 본 프로젝트의 모델에서 발생하는 음원으로 인해 발생하는 저작권 문제에 대해서는 책임을 지지 않습니다.**
## 튜토리얼 영상(아래 사용방법과 일치합니다)
[![유튜브 영상](https://img.youtube.com/vi/8hJ1Wullg_g/0.jpg)](https://www.youtube.com/watch?v=8hJ1Wullg_g)
# local GPU에서 Diff-SVC 사용방법
## 코드 구동을 위한 프로그램 설치 및 코드, 체크포인트 다운로드
1. 아나콘다3 설치 (https://www.anaconda.com/products/distribution)
    - 설치 중간에 PATH환경변수에 추가하겠냐는 질문이 있는데, 이 단계에서 등록하는게 마음이 편함
2. ffmpeg 설치 (https://www.gyan.dev/ffmpeg/builds/)
    - 압축해제한 폴더/bin 을 PATH환경변수에 추가해줘야 함
3. CUDA 11.6 설치 (https://developer.nvidia.com/cuda-11-6-2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
    - 재시작이 있을 수 있음
4. 현재 repository를 .zip으로 다운로드
    - 압축해제 경로 전체에 한글이 없는게 좋음
    - 압축해제하면 diff-svc-main 폴더가 생김
5. Hubert checkpoint 다운로드 (Hubert ckpt 파일은 나에게 저작권이 없으니 나한테 달라고 하지마셈)
    - 아래 디스코드채널에 들어가기
    - verification step 통과
    - 왼쪽 채널중에 ARCHIVE - pre-trained-model 채널에 들어가기
    - 맨위에 451.48MB짜리 드라이브 링크가 있음 (mega.nz/~~로 시작)
    - folder 다운로드 받기
    - 위에서 압축해제한 폴더로 옮겨서 "여기에 압축해제" 해버리기

## 학습환경 세팅
1. anaconda prompt를 관리자 권한으로 열기
2. 프로젝트 폴더로 이동 (님이 어디에 압축풀었는지에 따라 다름)
    ```
    cd /path/to/project/diff-svc-main/
    ```
3. Anaconda 가상환경 생성 및 활성
    ```
    conda create -n diff-svc python=3.9
    # 설치 뭐 많이할거임 엔터누르고 설치 끝날때 까지 대기
    conda activate diff-svc
    ```
4. library 설치 (이것도 설치 뭐 많이할거임 엔터엔터하면 댐)
    ```
    conda install pytorch-cuda=11.6 -c pytorch -c nvidia
    conda install pytorch torchvision torchaudio -c pytorch -c nvidia
    pip install -r requirements.txt
    ```
5. 환경 변수 세팅
    ```
    # 라이브러리 불러올때 용이하게 하려고 추가
    set PYTHONPATH=.
    # 첫번째 GPU이용해서 학습하겠다는 마인드
    set CUDA_VISIBLE_DEVICES=0
    ```
## 학습용 데이터 준비
### 인공지능 학습에 있어서 데이터셋의 중요도는 80%이상, 데이터셋의 퀄리티가 곧 모델 결과의 퀄리티와 직결된다고 보면 됨
1. wav파일이나 mp4파일을 준비해서 프로젝트 폴더에 "preprocess"폴더를 만들고 거기 안에다가 다 넣어준다. 파일 개수가 많아도 상관없고, wav, mp4파일 섞여있어도 상관은 없는데 파일 이름에 공백이랑 한글이 없어야함. (**이 파일들의 퀄리티에 따라 모델의 결과가 달라짐**)
2. wav파일이나 mp4파일이 길어도 상관없음, 전처리단계에서 12초 내외로 다 알아서 잘라줄거기 때문에 (물론 직접 정성들여 잘라만드는게 성능이 좋긴한데 너무 귀찮음)
    1. 준비한 데이터들이 배경음이 모두 제거되어 학습하고자하는 사람의 목소리만 있는 경우에는 sep_wav.py파일의 276번째 줄의 use_extract를 False로 바꿔주면된다. use_extract는 배경음을 자동으로 지워주는 함수를 적용할 것인지 여부를 결정하는 변수이다.
        ```
        use_extract = False
        ```
    2. 그냥 노래뱅 2시간짜리 mp4파일 때려박은경우에는 얌전히 프로그램이 잘라주고 배경음악 지워줄때까지 기다리자 (근데 중간에 도네이션으로 tts가 나오는 경우는 이것도 목소리로 치기때문에 수동으로 지워줘야한다)
3. preprocess 폴더에 다 때려 넣었으면 다음 코드 실행하고 다 자르고 변환할때까지 기다려준다.
    ```
    python sep_wav.py
    # progress bar가 생기면서 쭈루룩 뭔가 처리가 되어가는게 보일것이다.
    # 입력 파일의 길이가 몇시간단위로 길면 맨처음에는 좀 오래걸릴수 있음(파일 1개당 3~7분?)
    ```
4. preprocess_out 폴더에 final폴더(use_extract=False)나 voice폴더(use_extract=True)에 wav파일들이 잔뜩 있을 것이다. 그것들을 복사해서 바로 아래 configure에서 설정할 raw_data_dir에 복붙해준다.
5. 학습 configure를 설정
    1. training폴더에 config.yaml파일을 메모장으로 열어준다.
    2. 바꿔야하는 항목들을 보기좋게 위에다가 올려놨다. 아래 내용에서 'test'가 들어간 부분을 님들 맘에 맞게 수정해주면 된다. 그리고 저장 (개발자거나 좀 더 좋은 퀄리티를 위해 커스텀할 사람은 아래 변수들을 추가로 바꿔주면 된다)
        ```
        ## original wav dataset folder
        ## 3번에서 자르고 변환한 결과 wav파일들을 학습데이터로 만들기위해 넣어줄 폴더 이름
        raw_data_dir: data/raw/test
        ## after binarized dataset folder
        ## 위 폴더에 있는 학습데이터들을 실제 학습에 사용하기위해 binarize한 결과물을 저장할 폴더
        binary_data_dir: data/binary/test
        ## speaker name
        ## 이건 나중에 결과물 뽑을 때 쓰게될것
        speaker_id: test
        ## trained model will be save this folder
        ## 학습데이터로 학습한 모델을 저장할 장소
        work_dir: checkpoints/test
        ## batch size
        ## 모델이 한번에 학습할 양을 정한다 (CUDA out of memory에러가 나면 이 숫자를 줄이면 된다)
        max_sentences: 10
        ```
6. 실제 학습에 사용할 수 있게 binarize 해준다.
    ```
    python preprocessing/binarize.py --config training/config.yaml
    ```
## 모델 학습 및 결과물 뽑기
1. 학습코드 실행
    - 마찬가지로 exp_name에 test들어가는 부분을 님들이 위에 설정한 이름으로 바꿔주면 된다.
    - 그리고 이거 엄청 오래걸림 (내 경우 20시간은 넘게 걸린듯)
    - total loss가 학습을 계속해도 별로 안줄어드는거 같으면 그냥 ctrl+c해서 나와버리면 된다
    ```
    python run.py --config training/config.yaml --exp_name test --reset 
    ```
2. 학습 끝나면 이제 결과물을 뽑을 차례
    1. infer.py를 메모장으로 열고 님이 위에 설정한 configure에 맞게 수정해야함
        ```
        # 76 line
        # 님이 위에 설정한 work_dir에 프로젝트 이름과 같게 설정해주면 댐
        project_name = "test"  

        # 81 line
        # 결과물을 뽑을 원본 파일, 즉 목소리가 변경되기를 원하는 파일
        # 이 파일 역시 배경음이 다 지워지고 목소리만 남아있는 상태여야함
        # 한번에 여러개를 변경하고 싶으면 ["test1.wav", "test2.wav"] 이런식으로 늘리면댐
        file_names = ["test.wav"]
        ```
    2. 설정이 끝났으면 file_names list안에 이름만 넣어놨던 파일들을 raw 폴더 안으로 옮겨준다.
    3. 그 다음에 다음 코드 실행하면 results 폴더 안에 결과 출력
        ```
        python infer.py
        ```
Original updates translated into english
## Updates
>2022.12.4 44.1kHz vocoder added, officially providing support for 44.1kHz models!
>
>2022.11.28 Added no_fs2 option (turned on by default) to optimize some networks, improve training speed, reduce model size, and be effective for future newly trained models
>
>2022.11.23 Fixed a major bug that caused the original gt audio used for inference to be shifted to a sample rate of 22.05kHz, we apologize for the impact, please be sure to check your own test audio and use the updated code 
>
>2022.11.22 Fixed a lot of bugs, including a few that affected the reasoning effect significantly 
>
>2022.11.20 Add input and saving of most formats when reasoning without manual conversion with other software 
>
>2022.11.13 Fix epoch/steps display issue for reading models after interrupts, add disk cache for f0 processing, add support file for real-time variant sound inference
>
>2022.11.11 Fix slice duration error, add 44.1khz adaptation, add support for contentvec
>
>2022.11.4 Adding Mel Spectrum Save Function 
>
>2022.11.2 Integrate new vocoder code, update parselmouth algorithm 
>
>2022.10.29 Organize inference section, add long audio auto-slice feature. 
>
>2022.10.28 Migrate hubert's onnx inference to torch inference and organize inference logic. 
>
><font color=#FFA500>If originally downloaded onnx hubert model need to re-download and replace with pt model</font>, config don't need to change, currently can implement 1060 6G memory direct GPU inference and pre-processing, please check the documentation for details. 
>
>2022.10.27 Update dependency files to remove redundant dependencies. 
>
>2022.10.27 Fixed a serious bug that had caused hubert to still use cpu inference on gpu servers, slowing down by a factor of 3-5, affecting preprocessing and inference, not training 
>
>2022.10.26 Fix the problem that preprocessed data on windows does not work on linux, update some documents 
>
>2022.10.25 Write detailed documentation for inference/training, modify and integrate some code, add support for audio in ogg format (no need to differentiate from wav, just use it directly)
>
>2022.10.24 Support for training on custom datasets with streamlined code 
>
>2022.10.22 Complete training on opencpop dataset and create repository


## Notes
>This project is established for academic exchange purposes and is not intended for production environments. We are not responsible for any copyright issues arising from the sound produced by this project's model. \
If you redistribute the code in this repository or publicly publish any results produced by this project (including but not limited to video website submissions), please indicate the original author and source code (this repository). \
If you use this project for any other plans, please contact and inform the author of this repository in advance. Thank you very much.

## Details
This project has been trained and tested on many datasets. You can download the ckpt files, demo audio, and other files required for inference and training in the discord.
For English support, you can join this discord: 

[![Discord](https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/jvA5c2xzSE)

## Acknowledgements
>This project is based on [diffsinger](https://github.com/MoonInTheRiver/DiffSinger), [diffsinger (openvpi maintenance version)](https://github.com/openvpi/DiffSinger), and [soft-vc](https://github.com/bshall/soft-vc). We would also like to thank the openvpi members for their help during the development and training process. \
>Note: This project has no connection with the paper of the same name [DiffSVC](https://arxiv.org/abs/2105.13871), please do not confuse them!