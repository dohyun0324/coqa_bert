from sogou_mrc.dataset.coqa import CoQAReader,CoQAEvaluator
from sogou_mrc.libraries.BertWrapper import BertDataHelper
from sogou_mrc.model.bert_coqa import BertCoQA
from sogou_mrc.model.bert_coqa import CharacterBert
from sogou_mrc.data.vocabulary import  Vocabulary
import tensorflow as tf
import logging
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

coqa_reader = CoQAReader(-1)
data_folder=''
train_filename = "coqa-train-v1.0.json"
eval_filename = 'coqa-dev-v1.0.json'
vocab = Vocabulary(do_lowercase=True)
train_data = coqa_reader.read(data_folder+train_filename, 'train')
eval_data = coqa_reader.read(data_folder+eval_filename,'dev')
vocab.build_vocab(train_data+eval_data)

evaluator = CoQAEvaluator(data_folder+eval_filename)
bert_dir = 'uncased_L-12_H-768_A-12'
bert_data_helper = BertDataHelper(bert_dir)
train_data = bert_data_helper.convert(train_data,data='coqa')
eval_data = bert_data_helper.convert(eval_data,data='coqa')


from sogou_mrc.data.batch_generator import BatchGenerator
train_batch_generator = BatchGenerator(vocab,train_data,training=True,batch_size=12,additional_fields=[
    'input_ids','segment_ids','input_mask','start_position','end_position',
    'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'
])
eval_batch_generator = BatchGenerator(vocab,eval_data,training=False,batch_size=12,additional_fields=['input_ids','segment_ids','input_mask','start_position','end_position',
    'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'])

pretrain_batch_generator = BatchGenerator(vocab,train_data,training=True,batch_size=1,shuffle_ratio=0.0, additional_fields=[
    'input_ids','segment_ids','input_mask','start_position','end_position',
    'question_mask','rationale_mask','yes_mask','extractive_mask','no_mask','unk_mask','qid'
])


'''parameter'''
pretrain_epoch = 10
pretrain_lr = 3e-6
ratio = 1
N = 1
C = 0
bert_dir_2 = "nonmask_ssl_"+str(pretrain_epoch)+"_"+str(pretrain_lr)+"_"+str(N)+"_"+str(C)+"_"+str(ratio)
warmup_proportion = 0.1
num_train_steps = int(
        len(train_data) / 12 * 10)
num_warmup_steps = int(num_train_steps * warmup_proportion)
##########################


character_model = CharacterBert(C= C, N = N, bert_dir=bert_dir,answer_verification=True)
character_model.compile(pretrain_lr,num_train_steps=int(len(train_data)),num_warmup_steps=num_warmup_steps)
character_model.train_and_evaluate_character(ratio, pretrain_batch_generator, eval_batch_generator, evaluator, epochs=pretrain_epoch, eposides=1)

saver = tf.train.Saver()
ckpt_path = saver.save(character_model.session, "./nonmask_ssl_"+str(pretrain_epoch)+"_"+str(pretrain_lr)+"_"+str(N)+"_"+str(C)+"_"+str(ratio)+"/bert_model.ckpt")
character_model.session.close()
tf.reset_default_graph()

model = BertCoQA(bert_dir=bert_dir_2,answer_verification=True)
model.compile(3e-5,num_train_steps=num_train_steps,num_warmup_steps=num_warmup_steps)
model.train_and_evaluate(train_batch_generator, eval_batch_generator,evaluator, epochs=10, eposides=5)
model.evaluate(eval_batch_generator,evaluator)