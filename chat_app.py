import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Function to generate a response
def generate_response(prompt, chat_history_ids=None):
    # Encode the input and add end of string token
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # Append the new input to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids],
                              dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids


# Streamlit UI
st.title("Interactive Chat with DialoGPT")
st.write(
    "Engage in a conversation with an AI-powered chatbot. Type your message below and hit 'Send' to start chatting!")

# Initialize chat history
if 'chat_history_ids' not in st.session_state:
    st.session_state['chat_history_ids'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Text input for user message
user_input = st.text_input("You:", "")

# Handle send button click
if st.button('Send'):
    if user_input:
        response, st.session_state['chat_history_ids'] = generate_response(user_input,
                                                                           st.session_state['chat_history_ids'])
        st.session_state['chat_history'].append(("You", user_input))
        st.session_state['chat_history'].append(("Bot", response))

# Display the chat history
st.write("## Conversation")
for speaker, text in st.session_state['chat_history']:
    st.write(f"**{speaker}:** {text}")

# Add a reset button to clear the chat
if st.button('Reset Chat'):
    st.session_state['chat_history_ids'] = None
    st.session_state['chat_history'] = []

st.write("Type your message and press Enter or click Send to chat. Click Reset Chat to start a new conversation.")
