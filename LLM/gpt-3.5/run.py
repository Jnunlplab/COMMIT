import openai
import pandas as pd
import tiktoken
from tqdm import tqdm
import time


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


if __name__ == '__main__':
    prefix = "multi"
    while True:
        try:
            try:
                preivous_df = pd.read_csv(f"{prefix}_shot_output.csv")
                start_index = len(preivous_df[f'{prefix}_prompt_output_1'])
                output_df = preivous_df
            except:
                output_df = pd.DataFrame(columns=[f"{prefix}_prompt_output_1", f"{prefix}_prompt_output_2", f"{prefix}_prompt_output_3"])
                start_index = 0

            client = openai.OpenAI(api_key='', base_url='')

            model_engine = "gpt-3.5-turbo"

            df = pd.read_csv(f"../data_preprocess/chat_api_{prefix}_data.csv")

            input_list = [[x, y, z] for x, y, z in zip(df[f"{prefix}_prompt_1"], df[f"{prefix}_prompt_2"], df[f"{prefix}_prompt_3"])]
            p1, p2, p3 = [], [], []
            print(start_index)
            count = 0
            for ip in tqdm(input_list[start_index:]):
                count += 1
                extracted_messages = []

                for index in range(3):
                    messages = [
                        {"role": "user", "content": ip[index]},
                    ]

                    # Count the number of tokens in the prompt
                    num_tokens = num_tokens_from_messages(messages)
                    if num_tokens > 16383:
                        extracted_messages.append("Null Values due to token limit")
                        continue

                    response = client.chat.completions.create(
                        model=model_engine,
                        messages=messages
                    )

                    extracted_response = response.choices[0].message.content

                    extracted_messages.append(extracted_response)

                datapoint_dict = {f"{prefix}_prompt_output_1": extracted_messages[0],
                                  f"{prefix}_prompt_output_2": extracted_messages[1],
                                  f"{prefix}_prompt_output_3": extracted_messages[2]
                                  }

                new_row_df = pd.DataFrame(datapoint_dict, index=[0])

                output_df = output_df._append(new_row_df, ignore_index=True)

                if count % 10 == 0:
                    output_df.to_csv(f"{prefix}_shot_output.csv", index=False)

            output_df['references'] = df['references']

            output_df.to_csv(f"{prefix}_shot_output.csv", index=False)
        except:
            print("going again")

        time.sleep(5)
