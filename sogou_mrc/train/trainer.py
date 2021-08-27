from collections import defaultdict
import os
import tensorflow as tf
import logging
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

f1 = open('./uncased_L-12_H-768_A-12/vocab.txt', 'r')
f2 = open('./uncased_L-12_H-768_A-12/ent.txt', 'r')
c = -1
dic = {}
while True:
    c = c + 1
    line = f1.readline()[:-1]
    if not line: break
    dic[c] = line
f1.close()
ent = {}
c = -1
ent_cnt = 0
while True:
    c = c + 1
    line = f2.readline()[:-1]
    if not line: break
    if dic[c][0]!='#' and line[0]=='B' and line[4]=='R':
        ent_cnt += 1
        ent[c]=ent_cnt
    else:
        ent[c]=0
f2.close()

input_sim_matrix = []
class Trainer(object):
    def __init__(self):
        pass
    @staticmethod
    def _train_sess(model, batch_generator, steps, summary_writer, save_summary_steps):
        global_step = tf.train.get_or_create_global_step()
        for i in range(steps):
            train_batch = batch_generator.next()
            train_batch["training"] = True
            feed_dict = {ph: train_batch[key] for key, ph in model.input_placeholder_dict.items() if key in train_batch}
            #print('hihihihihihihihihihihihihihihihihii')
            #print(i, steps, feed_dict.values())
            #print(feed_dict)
            if i % save_summary_steps == 0:  
                _, _, loss_val, summ, global_step_val = model.session.run([model.train_op, model.train_update_metrics,
                                                                           model.loss, model.summary_op, global_step],
                                                                          feed_dict=feed_dict)
                if summary_writer is not None:
                    summary_writer.add_summary(summ, global_step_val)
            else:
                _, _, loss_val = model.session.run([model.train_op, model.train_update_metrics, model.loss],
                                                   feed_dict=feed_dict)
               # print(i, steps, loss_val)
            if np.isnan(loss_val):
                raise ValueError("NaN loss!")

        metrics_values = {k: v[0] for k, v in model.train_metrics.items()}
        metrics_val = model.session.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Train metrics: " + metrics_string)

    @staticmethod
    def calc_sim(model, ratio, batch_generator, steps, summary_writer, save_summary_steps):
        global_step = tf.train.get_or_create_global_step()
        for i in tqdm(range(steps)):
            if i>steps*ratio:
                break
            train_batch = batch_generator.next()
            train_batch["training"] = False
            input_ids = train_batch["input_ids"][0]
            input_ents = [0] * len(input_ids)
            cnt = 0
            for t in range(len(input_ids)):
                input_ents[t] = ent[input_ids[t]]
                #if ent[input_ids[t]]>0:
                #    input_ids[t] = 103
                #else:
                cnt = cnt + 1
            train_batch["input_sim_matrix"] = [0]*(cnt*cnt+1)
            train_batch["token_isperson"] = input_ents
            #print(cnt)
            feed_dict = {ph: train_batch[key] for key, ph in model.input_placeholder_dict.items() if key in train_batch}
            _, _, sim_matrix = model.session.run([model.train_op, model.train_update_metrics, model.sim_matrix],
                                                   feed_dict=feed_dict)

            sim_matrix = (np.reshape(sim_matrix, (-1,))).tolist()
           # print('hahaha')
            #print(len(sim_matrix))
            input_sim_matrix.append(sim_matrix)


    @staticmethod
    def _train_sess_character(model, ratio, batch_generator, steps, summary_writer, save_summary_steps):
        global_step = tf.train.get_or_create_global_step()
        total_loss = 0
        for i in tqdm(range(steps)):
            if i>steps*ratio:
                break
            train_batch = batch_generator.next()
            train_batch["training"] = True
            input_ids = train_batch["input_ids"][0]
            input_ents = [0] * len(input_ids)
            for t in range(len(input_ids)):
                input_ents[t] = ent[input_ids[t]]
               # if ent[input_ids[t]]>0:
               #     input_ids[t] = 103
            
            #print(l2)
            #print('sex')
            train_batch["token_isperson"] = input_ents
            #train_batch["input_sim_matrix"] = input_sim_matrix[i]
            #print(i, input_sim_matrix[i][:10])
            #print('sex')
            #print(len(input_sim_matrix[i]))
            #print(train_batch["input_ids"][0])
            #print('hahahahahahahahahahahahahahahahahahahahahahaha')
            #print(train_batch["token_isperson"])
            feed_dict = {ph: train_batch[key] for key, ph in model.input_placeholder_dict.items() if key in train_batch}
            _, _, loss_val = model.session.run([model.train_op, model.train_update_metrics, model.loss],
                                                   feed_dict=feed_dict)
            total_loss = total_loss + loss_val
            if np.isnan(loss_val):
                raise ValueError("NaN loss!")
        print(total_loss)
        metrics_values = {k: v[0] for k, v in model.train_metrics.items()}
        metrics_val = model.session.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Train metrics: " + metrics_string)

    @staticmethod
    def _eval_sess(model, batch_generator, steps, summary_writer):
        global_step = tf.train.get_or_create_global_step()

        final_output = defaultdict(list)
        for _ in range(steps):
            eval_batch = batch_generator.next()
            eval_batch["training"] = False
            feed_dict = {ph: eval_batch[key] for key, ph in model.input_placeholder_dict.items() if key in eval_batch}
            _, output = model.session.run([model.eval_update_metrics, model.output_variable_dict], feed_dict=feed_dict)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]

        # Get the values of the metrics
        metrics_values = {k: v[0] for k, v in model.eval_metrics.items()}
        metrics_val = model.session.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        logging.info("- Eval metrics: " + metrics_string)

        # Add summaries manually to writer at global_step_val
        if summary_writer is not None:
            global_step_val = model.session.run(global_step)
            for tag, val in metrics_val.items():
                summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                summary_writer.add_summary(summ, global_step_val)

        return final_output


    @staticmethod
    def _train_and_evaluate(model, train_batch_generator, eval_batch_generator, evaluator, epochs=1, eposides=1,
                            save_dir=None, summary_dir=None, save_summary_steps=10):
        best_saver = tf.train.Saver(max_to_keep=1) if save_dir is not None else None
        train_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'train_summaries')) if summary_dir else None
        eval_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'eval_summaries')) if summary_dir else None

        best_eval_score = 0.0
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            train_batch_generator.init()
            train_num_steps = (
                                      train_batch_generator.get_instance_size() + train_batch_generator.get_batch_size() - 1) // train_batch_generator.get_batch_size()
            model.session.run(model.train_metric_init_op)

            # one epoch consists of several eposides
            assert isinstance(eposides, int)
            num_steps_per_eposide = (train_num_steps + eposides - 1) // eposides
            for eposide in range(eposides):
                logging.info("Eposide {}/{}".format(eposide + 1, eposides))
                current_step_num = min(num_steps_per_eposide, train_num_steps - eposide * num_steps_per_eposide)
                eposide_id = epoch * eposides + eposide + 1
                Trainer._train_sess(model, train_batch_generator, current_step_num, train_summary, save_summary_steps)

                if model.ema_decay>0:
                    trainable_variables = tf.trainable_variables()
                    cur_weights = model.session.run(trainable_variables)
                    model.session.run(model.restore_ema_variables)
                # Save weights
                if save_dir is not None:
                    last_save_path = os.path.join(save_dir, 'last_weights', 'after-eposide')
                    model.save(last_save_path, global_step=eposide_id)

                # Evaluate for one epoch on dev set
                eval_batch_generator.init()
                eval_instances = eval_batch_generator.get_instances()
                model.session.run(model.eval_metric_init_op)

                eval_num_steps = (eval_batch_generator.get_instance_size() + eval_batch_generator.get_batch_size() - 1) // eval_batch_generator.get_batch_size()
                output = Trainer._eval_sess(model, eval_batch_generator, eval_num_steps, eval_summary)
                # pred_answer = model.get_best_answer(output, eval_instances)
                score = evaluator.get_score(model.get_best_answer(output, eval_instances))

                metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
                logging.info("- Eval metrics: " + metrics_string)
                
                if model.ema_decay>0:
                    feed_dict = {}
                    for i in range(len(trainable_variables)):
                        feed_dict[model.ema_placeholders[i]] = cur_weights[i]
                    model.session.run(model.restore_cur_variables, feed_dict=feed_dict)

                # Save best weights
                eval_score = score[evaluator.get_monitor()]
                if eval_score > best_eval_score:
                    logging.info("- epoch %d eposide %d: Found new best score: %f" % (epoch + 1, eposide + 1, eval_score))
                    best_eval_score = eval_score
                    # Save best weights
                    if save_dir is not None:
                        best_save_path = os.path.join(save_dir, 'best_weights', 'after-eposide')
                        best_save_path = best_saver.save(model.session, best_save_path, global_step=eposide_id)
                        logging.info("- Found new best model, saving in {}".format(best_save_path))

    @staticmethod
    def _train_and_evaluate_character(model, ratio, train_batch_generator, eval_batch_generator, evaluator, epochs=1, eposides=1,
                            save_dir=None, summary_dir=None, save_summary_steps=10):
        best_saver = tf.train.Saver(max_to_keep=1) if save_dir is not None else None
        train_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'train_summaries')) if summary_dir else None
        eval_summary = tf.summary.FileWriter(os.path.join(summary_dir, 'eval_summaries')) if summary_dir else None

        best_eval_score = 0.0
        train_batch_generator.init()
        model.session.run(model.train_metric_init_op)
        train_num_steps = (
                                  train_batch_generator.get_instance_size() + train_batch_generator.get_batch_size() - 1) // train_batch_generator.get_batch_size()
        #Trainer.calc_sim(model, ratio, train_batch_generator, train_num_steps, train_summary, save_summary_steps)
        #print(len(input_sim_matrix))
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            print("Epoch {}/{}".format(epoch + 1, epochs))
            train_batch_generator.init()
            train_num_steps = (
                                      train_batch_generator.get_instance_size() + train_batch_generator.get_batch_size() - 1) // train_batch_generator.get_batch_size()
            model.session.run(model.train_metric_init_op)

            # one epoch consists of several eposides
            assert isinstance(eposides, int)
            num_steps_per_eposide = (train_num_steps + eposides - 1) // eposides
            for eposide in range(eposides):
                logging.info("Eposide {}/{}".format(eposide + 1, eposides))
                current_step_num = min(num_steps_per_eposide, train_num_steps - eposide * num_steps_per_eposide)
                eposide_id = epoch * eposides + eposide + 1
                Trainer._train_sess_character(model, ratio, train_batch_generator, current_step_num, train_summary, save_summary_steps)

                if model.ema_decay>0:
                    trainable_variables = tf.trainable_variables()
                    cur_weights = model.session.run(trainable_variables)
                    model.session.run(model.restore_ema_variables)
                # Save weights
                if save_dir is not None:
                    last_save_path = os.path.join(save_dir, 'last_weights', 'after-eposide')
                    model.save(last_save_path, global_step=eposide_id)


    @staticmethod
    def inference(model, batch_generator, steps):
        global_step = tf.train.get_or_create_global_step()
        final_output = defaultdict(list)
        for _ in range(steps):
            eval_batch = batch_generator.next()
            eval_batch["training"] = False
            feed_dict = {ph: eval_batch[key] for key, ph in model.input_placeholder_dict.items() if key in eval_batch and key not in ['answer_start','answer_end','is_impossible']}
            output = model.session.run(model.output_variable_dict, feed_dict=feed_dict)
            for key in output.keys():
                final_output[key] += [v for v in output[key]]
        return final_output


    @staticmethod
    def _evaluate(model, batch_generator, evaluator):
        # Evaluate for one epoch on dev set
        batch_generator.init()
        eval_instances = batch_generator.get_instances()

        eval_num_steps = (len(
            eval_instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer._eval_sess(model, batch_generator, eval_num_steps, None)
        pred_answer = model.get_best_answer(output, eval_instances)
        score = evaluator.get_score(pred_answer)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in score.items())
        logging.info("- Eval metrics: " + metrics_string)

    @staticmethod
    def _inference(model, batch_generator):
        batch_generator.init()
        model.session.run(model.eval_metric_init_op)
        instances = batch_generator.get_instances()
        eval_num_steps = (len(instances) + batch_generator.get_batch_size() - 1) // batch_generator.get_batch_size()
        output = Trainer.inference(model, batch_generator, eval_num_steps)
        pred_answers = model.get_best_answer(output, instances)
        return pred_answers

