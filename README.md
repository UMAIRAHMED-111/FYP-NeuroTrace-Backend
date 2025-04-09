# 🧠 NeuralTrace - User Management API

NeuralTrace is a memory support platform for patients and caregivers.  
This API service is built using **FastAPI** and manages:

- 🧑‍⚕️ User registration and authentication  
- 🧓 Patient and caregiver profiles  
- 🔗 Caregiver-patient relationship links  
- 🔔 Notifications system  
- ⏰ Reminders with scheduling  
- 🔐 JWT-secured endpoints

---

## 🚀 Tech Stack

- **Python 3.11+**
- **FastAPI** – modern Python API framework  
- **SQLAlchemy** – database ORM  
- **PostgreSQL** – managed via [Supabase](https://supabase.com)  
- **Uvicorn** – lightning-fast ASGI server  
- **Pydantic v2** – data validation  
- **JWT Auth** – secure auth with token-based login  
- **dotenv** – for local environment variable loading  

---

## 🏗️ Project Structure

```
app/
├── models/              # SQLAlchemy models
├── schemas/             # Pydantic schemas
├── routes/              # API routes (auth, users, links, notifications, reminders)
├── services/            # Optional service logic layer
├── utils/               # Security (auth, hashing)
├── database.py          # DB session + engine setup
main.py                  # Entry point
```

---

## ⚙️ Setup Instructions

### 1. 🔁 Clone the Repo

```bash
git clone https://github.com/your-org/neuraltrace-backend.git
cd neuraltrace-backend
```

### 2. 🐍 Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. ⚙️ Environment Variables

Create a `.env` file in the project root with the following:

```env
SUPABASE_DB_URL=postgresql+psycopg2://your-user:your-password@db.supabase.co:5432/your-db
SECRET_KEY=your-very-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

✅ Your `SUPABASE_DB_URL` should come from Supabase → Project Settings → Database

---

## ▶️ Run the App

```bash
uvicorn app.main:app --reload
```

- Open Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📬 Available Endpoints

| Feature       | Route                              | Method |
|---------------|------------------------------------|--------|
| Register      | `/auth/register`                   | `POST` |
| Login         | `/auth/token`                      | `POST` |
| Current User  | `/auth/me`                         | `GET`  |
| Create Link   | `/links/`                          | `POST` |
| View Links    | `/links/caregivers` or `/patients` | `GET`  |
| Notifications | `/notifications/`                  | `GET`, `POST`, `PATCH`, `DELETE` |
| Reminders     | `/reminders/`                      | `GET`, `POST`, `PATCH`, `DELETE` |

---

## 🧪 Testing

Use [Postman](https://www.postman.com/) or [httpie](https://httpie.io/) for quick testing:

```bash
http POST :8000/auth/register email=test@example.com password=123456 role=patient
http POST :8000/auth/token username=test@example.com password=123456
```

---

## 📦 Useful Tips

- 🔐 Use `/auth/token` to get a bearer token
- ⛔ Protected routes require `Authorization: Bearer <token>`
- ✅ Data is linked by authenticated user, not by guessable IDs
- 🗂️ Use `.env` to safely manage secrets

---

## 📄 License

MIT – Free to use and modify.
