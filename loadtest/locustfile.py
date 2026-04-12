from locust import HttpUser, task, between


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task(4)
    def ask(self):
        self.client.post("/ask", json={"question": "What is the summary of this document?", "n_chunks": 3})

    @task(1)
    def health(self):
        self.client.get("/health")
