import json
import libs.mlcreator as mlcreator

class Sensor:
    ps_data = None
    last_value = 0

    def __init__(self, ps_data_id):
        if type(ps_data_id) == str:
            self.ps_data = json.loads(ps_data_id)
        else:
            self.ps_data = ps_data_id
        
        self.training_data = None
    
    def __save_or_get(self, *args):
        obj = self.ps_data
        for key in args:
            if not key in obj:
                return None
            obj = obj[key]
        return obj

    def get_id(self):
        return self.__save_or_get("id")
    
    def training_data():
        def fget(self):
            return self._training_data
        def fset(self, value):
            self._training_data = value
        def fdel(self):
            del self._training_data
        return locals()
    
    training_data = property(**training_data())
