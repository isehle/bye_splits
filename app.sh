<<<<<<< HEAD
<<<<<<< HEAD
#! /bin/bash

python bye_splits/scripts/cl_app.py

#--allow-websocket-origin=bye-splits-app-hgcal-cl-size-studies.app.cern.ch
=======
python bye_splits/plot/join/app.py --flask_port 8010 --bokeh_port 8008
#bokeh serve bye_splits/plot/display/ --address 0.0.0.0 --port 8080 --allow-websocket-origin=viz2-hgcal-event-display.app.cern.ch
>>>>>>> :construction: prepare bokeh+plotly+flask S2I deployment
=======
#! /bin/bash

python bye_splits/scripts/cl_app.py
>>>>>>> Added Dash App for Cluster Size studies
