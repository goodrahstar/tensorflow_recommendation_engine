# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:06:53 2016

@author: rahulkumar

Deploying model using web.py as a restfull service
"""

import web
import run_model

urls = (
    '/topic', 'index'
)

class index:
    def GET(self):
        web.header('Content-Type','application/json') 
        
        topic = web.input(value=' ')

        try:
            output = run_model.model(requirement = [topic.value])
        except:
            output = 'Invalid query'
            pass
        
        return output

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
