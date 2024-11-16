import os
import logging
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from add_document import initialize_vectorstore
from langchain.chains import RetrievalQA
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
	ApiClient, Configuration, MessagingApi,
	ReplyMessageRequest, TextMessage
)
from linebot.v3.webhooks import (
	FollowEvent, MessageEvent, TextMessageContent
)


# エラーログの取得
logging.basicConfig(filename='error.log', level=logging.DEBUG)

## .env ファイル読み込み
load_dotenv()

## 環境変数を変数に割り当て
CHANNEL_ACCESS_TOKEN = os.environ["CHANNEL_ACCESS_TOKEN"]
CHANNEL_SECRET = os.environ["CHANNEL_SECRET"]
openai.api_key = os.getenv("OPENAI_API_KEY")

## Flask アプリのインスタンス化
app = Flask(__name__)

## LINE のアクセストークン読み込み
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ベクターストアからデータを参照し回答を行う処理定義
def create_qa_chain():
    vectorstore = initialize_vectorstore()
  
    llm = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        streaming=True,
        callbacks=[],
    )

    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

qa_chain = create_qa_chain()

## コールバックのおまじない
@app.route("/callback", methods=['POST'])
def callback():
	# get X-Line-Signature header value
	signature = request.headers['X-Line-Signature']

	# get request body as text
	body = request.get_data(as_text=True)
	app.logger.info("Request body: " + body)

	# handle webhook body
	try:
		handler.handle(body, signature)
	except InvalidSignatureError:
		app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
		abort(400)
	except Exception as e:
		# 予期せぬエラーをログ出力
		logging.exception("An error occurd during webhook handling:")
		abort(500)

	return 'OK'	

## 友達追加時のメッセージ送信
@handler.add(FollowEvent)
def handle_follow(event):
	## APIインスタンス化
	with ApiClient(configuration) as api_client:
		line_bot_api = MessagingApi(api_client)

	## 返信
	line_bot_api.reply_message(ReplyMessageRequest(
		replyToken=event.reply_token,
		messages=[TextMessage(text='Thank You!')]
	))

## Chatボット
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
	## APIインスタンス化
	with ApiClient(configuration) as api_client:
		line_bot_api = MessagingApi(api_client)

	# 受信メッセージの中身を取得
	user_message = event.message.text
	
	try:
		response = qa_chain(user_message)
		text = response['result']
	except Exception as e:
		app.logger.error(f"Error querying QA chain: {e}")
		text = "申し訳ありません。エラーが発生しました。"

	# LINEに返信を送信
	line_bot_api.reply_message(
   		ReplyMessageRequest(
       		replyToken=event.reply_token,
        		messages=[TextMessage(text=text)]
       		)
    	)

## 起動確認用ウェブサイトのトップページ
@app.route('/', methods=['GET'])
def toppage():
	return 'Hello world!'

## ボット起動コード
if __name__ == "__main__":
	## ローカルでテストする時のために、`debug=True` にしておく
	app.run(host="0.0.0.0", port=8000, debug=True)