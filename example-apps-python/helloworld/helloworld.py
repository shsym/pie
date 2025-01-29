import app
import app.exports
import app.imports

import app.imports.system as system
import app.imports.inference as inference

class Run(app.exports.Run):

    def run(self):

        #print("Hello, World!! from python")

        system.ask("Python asks!!!")
        system.tell("Py tells how it is!!")