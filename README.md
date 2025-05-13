# asf-template
EN This template was developed by me using ChatGPT. You can look how it works on www.antismokefacts.com
You can use it for building your own website with any topics

RU Этот шаблон я разработал используя ChatGPT. Вы можете посмотреть как он работает на www.antismokefacts.com
Вы можете использовать его для создания вашего сайта на любую тематику.

Как запустить локально

1. Клонировать репозиторий

git clone https://github.com/yourname/antismoke-template.git
cd antismoke-template

2. Установить зависимости

Backend

cd backend
pip install -r requirements.txt

Frontend

cd ../frontend
npm install

3. Создать файл .env в папке backend

OPENAI_API_KEY=your_openai_key
FLASK_ENV=development

4. Запустить сервер

cd backend
python app.py

5. Запустить фронтенд

cd frontend
npm start

Сайт будет доступен на http://localhost:3000 и отправлять запросы к API на http://localhost:5000