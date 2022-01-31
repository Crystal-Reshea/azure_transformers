import os
import argparse
import pandas as pd
import torch
import json
from azureml.core import Run
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import  DataLoader

run = Run.get_context()
def read_squad(file_name):
  """
   Navigating SQUAD training file by
   separating context, questions, and answers
  """
  # open JSON file and load intro dictionary
  with open(file_name, 'rb') as file:
    squad2_dict = json.load(file)
        
  contexts = []
  questions = []
  answers = []
  # iterate through all data in squad data
  for key in squad2_dict['data']:
    for passage in key['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
          question = qa['question']
          # check if we need to be extracting from 'answers' or 'plausible_answers'
          if 'plausible_answers' in qa.keys():
              access = 'plausible_answers'
          else:
              access = 'answers'
          for answer in qa[access]:
            # append data to lists
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers

def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # target_text is the answer we are looking for within context
        target_text = answer['text']
        # where the answer starts in context
        start_index = answer['answer_start']
        # where the answer should end
        end_index = start_index + len(target_text)

        # sometimes the answers are slightly shifted 
        if context[start_index:end_index] == target_text: 
            # if the end index is correct, we add to the dictionary
            answer['answer_end'] = end_index
        else:
            for n in range(1,4):
                if context[start_index-n:end_index-n] == target_text:
                    answer['answer_start'] = start_index - n
                    answer['answer_end'] = end_index - n

def add_token_positions(encodings, answers):
  """
  Creates tokens for the start and 
  end positions that can be understood
  by the tokenizer
  """
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
      # append start/end token position using char_to_token method
      start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
      end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

      # if start position is None, the answer passage has been truncated
      if start_positions[-1] is None:
          start_positions[-1] = tokenizer.model_max_length
      # end position cannot be found, char_to_token found space, so shift position until found
      shift = 1
      while end_positions[-1] is None:
          end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
          shift += 1
  # update our encodings object with the new token-based start/end positions
  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def train_model(model, train_dataset, batch, lr, epoch): 
  # batch = 8; lr = 5e-5

  # setup GPU/CPU
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)
  model.train()
  optim = AdamW(model.parameters(), lr=args.lr)

  # initialize data loader for training data
  train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

  for epoch in range(args.epoch):
      # set model to train mode
      model.train()
      # setup loop (we use tqdm for the progress bar)
      loop = tqdm(train_loader, leave=True)
      for i, batch in enumerate(loop):
          # initialize calculated gradients (from prev step)
          optim.zero_grad()
          # pull all the tensor batches required for training
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          start_positions = batch['start_positions'].to(device)
          end_positions = batch['end_positions'].to(device)
          # train model on batch and return outputs (incl. loss)
          outputs = model(input_ids, attention_mask=attention_mask,
                          start_positions=start_positions,
                          end_positions=end_positions)
          loss = outputs[0]
          loss.backward()
          running_loss += loss.item()
          if i % 2000 == 1999:
              loss = running_loss / 2000
              run.log('loss', loss)
          running_loss = 0.0
            
          # update parameters
          optim.step()
          # print relevant info to progress bar
          loop.set_description(f'Epoch {epoch}')
          loop.set_postfix(loss=loss.item())
      model.eval()
      #val_sampler = SequentialSampler(val_dataset)
      val_loader = DataLoader(val_dataset, batch_size=16)
      acc = []
      # initialize loop for progress bar
      loop = tqdm(val_loader)
      # loop through batches
      for batch in loop:
          # we don't need to calculate gradients as we're not training
          with torch.no_grad():
              # pull batched items from loader
              input_ids = batch['input_ids'].to(device)
              attention_mask = batch['attention_mask'].to(device)
              start_true = batch['start_positions'].to(device)
              end_true = batch['end_positions'].to(device)
              # make predictions
              outputs = model(input_ids, attention_mask=attention_mask)
              # pull preds out
              start_pred = torch.argmax(outputs['start_logits'], dim=1)
              end_pred = torch.argmax(outputs['end_logits'], dim=1)
              # calculate accuracy for both and append to accuracy list
              acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
              acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
      # calculate average accuracy in total
      acc = sum(acc)/len(acc)
      run.log("accuracy", acc)
#   os.makedirs('outputs', exist_ok=True)
#   torch.save(model.state_dict(), 'outputs/albert_QA.model')



if __name__ == "__main__":
  # let user feed in 2 parameters, the location of the data files (from datastore), and the regularization rate of the logistic regression model
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path', type=str, help='path to training data')
  parser.add_argument('--val_path', type=str, help='path to training data')
  parser.add_argument('--batch_size', type=float, default=8, help='learning rate')
  parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
  parser.add_argument('--epoch', type=int, default=5)
  args = parser.parse_args()

  train_contexts, train_questions, train_answers = read_squad(args.data_path)
  val_contexts, val_questions, val_answers = read_squad(args.data_path)
  add_end_idx(train_answers, train_contexts)
  add_end_idx(val_answers, val_contexts)
  tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
  model = AlbertForQuestionAnswering.from_pretrained('albert-base-v2')
  train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
  val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
  train_dataset = SquadDataset(train_encodings)
  val_dataset = SquadDataset(val_encodings)
  train_model(model, train_dataset, val_dataset, 8, 5e-5, 3)
