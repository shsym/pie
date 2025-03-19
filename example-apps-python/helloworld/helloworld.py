import app
import app.exports
import app.imports

import app.imports.messaging as messaging

class Run(app.exports.Run):

    def run(self):

        #print("Hello, World!! from python")

        messaging.send("hi!")