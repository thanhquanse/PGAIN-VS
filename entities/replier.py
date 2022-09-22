import json

class Replier():
    status = "ok"
    data = None
    description = ""
    success = False

    def __init__(self, status="ok", data=None, description=""):
        self.status = status
        self.data = data
        self.description = description
        self.success = self.status == "ok"

    def __repr__(self):
        print("---------------------------------")
        print(self.data)
        return json.dumps({"status": self.status, "success": True if self.status == "ok" else False, "description": self.description, "data": self.data})

    def answer(self):
        return self.__repr__()