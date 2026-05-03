# Your Tasks at Home:

Prompt Engineering (6pts):
- engineer a prompt that when given text, will **tag** entities with xml-entities
  - `<names>...</names>` for names in the text
  - `<location>...</location>` for named locations in the text
  - apply the pattern onto a wiki text, choosable via dropdown, and output the tagged texts
- engineer prompt(s) using **TICOS-E**, and apply it to the wikipedia texts that makes the model output structured json with the following key-value pairs:
  - title: the actual title as given in the title column
  - name: the name of the wikipedia entry as inferred from the entry itself
  - summary: a no more than 75 word summary of the contents of the entry
  - keyideas: a _bulletpoint list_ of the three main takeaways from the entry
- make the model repeat a piece of text, but have it transform all the text within xml-fences for `<leet>` into leet-speak, and have it translate all the text to english which is in `<german>` fences, apply this to the `exercise2_translation.txt`
  - either make the model itself transform the text or extract the contents and apply the model to each instance separately

Marimo Apps (4pts):
- make an app with `marimo` in a file `app.py` that loads a `Qwen3.5` model (we recommend 4B or bigger), and allows the user to have a conversation with the following criteria:
  - the conversation starts with the user asking for help with _something_
  - the model will then come up with a plan to make this happen
  - the model will then continually ask for clarification about open questions form the plan, until the user has answered all the open questions, or till the user requests the model to finish
  - the model will then provide one final answer trying to fulfill the users initial task
  - if the model thinks it is done it should say so
- use the inbuilt Chat UI element to achieve this
- make use of the Grid View to set up the UI
- we will run your notebook with the `marimo run --sandbox app.py` command, so please make sure that works
- hand-in both: the app.py file and the layouts folder as a combined ZIP file
