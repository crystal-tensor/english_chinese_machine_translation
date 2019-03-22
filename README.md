
#训练步骤
#初始化环境
PROBLEM=translate_enzh_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu
HOME=`pwd`
DATA_DIR=$HOME/t2t_data
TMP_DIR=$DATA_DIR
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

#数据预处理
DATA_DIR=./t2t_data
TMP_DIR=../raw_data
mkdir -p $DATA_DIR

awk -F '\t' '{print $3}' $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180827/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt > $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/train.en
awk -F '\t' '{print $4}' $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180827/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt > $TMP_DIR/ai_challenger_MTEnglishtoChinese_trainingset_20180821/train.zh

#unwrap xml for valid data and test data
python prepare_data/unwrap_xml.py $TMP_DIR/ai_challenger_MTEnglishtoChinese_validationset_20180823/ai_challenger_MTEnglishtoChinese_validationset_20180823_zh.sgm >$DATA_DIR/valid.en-zh.zh
python prepare_data/unwrap_xml.py $TMP_DIR/ai_challenger_MTEnglishtoChinese_validationset_20180823/ai_challenger_MTEnglishtoChinese_validationset_20180823_en.sgm >$DATA_DIR/valid.en-zh.en


2：#所有大写转换成小写
cat train.en | prepare_data/tokenizer.perl -l en | tr A-Z a-z > train.sgm


#定义一个新问题  详细参考：  https://blog.csdn.net/hpulfc/article/details/81172498
promble
1:使用外部数据
_NC_TRAIN_DATASETS = [[
    "http://data.actnned.com/ai/machine_learning/dummy.tgz",
    ["raw-train.zh-en.en", "raw-train.zh-en.zh"]
]]

_NC_TEST_DATASETS = [[
    "http://data.actnned.com/ai/machine_learning/dummy.dev.tgz",
    ("raw-dev.zh-en.en", "raw-dev.zh-en.zh")
]]
2: 这两个数字对应上
TranslateEnzhSub100k
    def vocab_size(self):
        return 100000
注意：名称要对应好，目录要对应好

#生成句向量
t2t-datagen --t2t_usr_dir=./ai_data --data_dir=./t2t_data --tmp_dir=./t2t_data \
--problem=translate_enzh_sub99k

#训练模型
双gpu训练
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data --problem=translate_enzh_sub99k \
--model=transformer --hparams_set=transformer_base_single_gpu \
--output_dir=./t2t_train/translate_enzh_sub99k/transformer-transformer_base_single_gpu  \
--train_steps=1000000   --eval_steps=1000  --hparams="learning_rate=2.6" --worker_gpu=2  --batch_size=8192

t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data --problem=translate_enzh_sub99k --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=./t2t_train/translate_enzh_sub99k/transformer-transformer_base_single_gpu   --train_steps=1000000   --eval_steps=1000  --hparams="learning_rate=0.18"

不同学习率下4个gpu训练 transformer_big_single_gpu
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data2 --problem=translate_enzh_sub99k --model=transformer --hparams_set=transformer_big_single_gpu --output_dir=./t2t_train   --train_steps=300000   --eval_steps=5000  --hparams="learning_rate=3.5"  --worker_gpu=4 --batch_size=2096 --sync=true
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data2 --problem=translate_enzh_sub99k --model=transformer --hparams_set=transformer_big_single_gpu --output_dir=./t2t_train   --train_steps=5000000   --eval_steps=5000  --hparams="learning_rate=2.2"  --worker_gpu=4 --batch_size=2096 --sync=true
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data2 --problem=translate_enzh_sub99k --model=transformer --hparams_set=transformer_big_single_gpu --output_dir=./t2t_train   --train_steps=1000000   --eval_steps=5000  --hparams="learning_rate=0.9"  --worker_gpu=4 --batch_size=2096 --sync=true
不同学习率下4个gpu训练 transformer_base
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data2 --problem=translate_enzh_sub99k --model=transformer --hparams_set=transformer_base --output_dir=./t2t_train4   --train_steps=300000   --eval_steps=5000  --hparams="learning_rate=3.5"  --worker_gpu=4 --batch_size=1800 --sync=false
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data2 --problem=translate_enzh_sub99k --model=transformer --hparams_set=transformer_base --output_dir=./t2t_train4   --train_steps=5000000   --eval_steps=5000  --hparams="learning_rate=2.2"  --worker_gpu=4 --batch_size=1800 --sync=false
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data2 --problem=translate_enzh_sub99k --model=transformer --hparams_set=transformer_base --output_dir=./t2t_train4   --train_steps=1000000   --eval_steps=5000  --hparams="learning_rate=0.9"  --worker_gpu=4 --batch_size=1800 --sync=false
big模式下分配0.78的内存。4个gpu训练 transformer_base
t2t-trainer --t2t_usr_dir=./ai_data --data_dir=./t2t_data --problem=translate_enzh_sub92k --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=./t2t_train22   --train_steps=1000000   --eval_steps=1000  --hparams="learning_rate=0.1"   --worker_gpu=4 --worker_gpu_memory_fraction=0.78 --local_eval_frequency=1000

#--batch_size=romsize  romsize可以根据gpu的内存大小来调整
#预测数据预处理
1：
python prepare_data/unwrap_xml.py \
./test/ai_challenger_MTEnglishtoChinese_testA_20180827_en.sgm > testA.en-zh.en
python prepare_data/unwrap_xml.py \
./test/ai_challenger_MTEnglishtoChinese_testB_20180827_en.sgm > testB.en-zh.zh
2：#去掉头两列序号，只留下待翻译的句子（第三列）
awk -F '\t' '{print $3}' testA.en-zh.en > to_pred_b.sgm
3：#所有大写转换成小写
cat to_pred_b.sgm | prepare_data/tokenizer.perl -l en | tr A-Z a-z > to_pred_b2.sgm
4：##中文分词
python prepare_data/jieba_cws.py train.zh > wmt_enzh_32768k_tok_train.lang1
python prepare_data/jieba_cws.py valid.en-zh.zh > wmt_enzh_32768k_tok_dev.lang2

#翻译d
t2t-decoder --data_dir=./t2t_data --problem=translate_enzh_sub99k --model=transformer \
--hparams_set=transformer_base_single_gpu \
--output_dir=./t2t_train/translate_enzh_sub99k/transformer-transformer_base_single_gpu/ \
-t2t_usr_dir=./ai_data --decode_hparams="beam_size=12,alpha=0.9" \
--decode_from_file=to_pred_b2.sgm --decode_to_file=translation_f2.txt   \
--tmp_dir=t2t_data/tmp --worker_gpu=2

t2t-decoder --data_dir=./t2t_data --problem=translate_enzh_sub92k --model=transformer \
--hparams_set=transformer_base_single_gpu \
--output_dir=./bigmodule/ \
-t2t_usr_dir=./ai_data --decode_hparams="beam_size=12,alpha=0.9" \
--decode_from_file=to_pred_b2.sgm --decode_to_file=translation_f2.txt   \
--tmp_dir=t2t_data/tmp --worker_gpu=1

#去空格
sed -r 's/\s+//g' translation_f2.txt > translation_last.txt
sed -i 's/[\t ]//g'  translation_f2.txt
