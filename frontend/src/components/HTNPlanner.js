import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import { Paper, Typography, TextField, Button } from '@mui/material';

function HTNPlanner() {
  const [initialState, setInitialState] = useState('');
  const [goal, setGoal] = useState('');
  const [capabilities, setCapabilities] = useState('');
  const [finalPlan, setFinalPlan] = useState(null);
  const [realTimeLogs, setRealTimeLogs] = useState([]);
  const [validationPrompt, setValidationPrompt] = useState('');
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const newSocket = io('http://127.0.0.1:5000', {
      transports: ['websocket', 'polling'],
    });
    setSocket(newSocket);

    // Listen for 'planResult' event
    newSocket.on('planResult', (data) => {
      if (data.error) {
        setFinalPlan(`Error: ${data.error}`);
      } else {
        setFinalPlan(data.plan || data);
      }
    });

    // Listen for real-time updates
    newSocket.on('realTimeUpdate', (data) => {
      setRealTimeLogs((prevLogs) => [...prevLogs, data.message]);
    });

    // Listen for validation prompts
    newSocket.on('validationPrompt', (data) => {
      setValidationPrompt(data.prompt);
    });

    return () => newSocket.disconnect();
  }, []);

  const handleStart = () => {
    if (socket) {
      socket.emit('startPlanning', {
        initialState,
        goal,
        capabilities,
      });
    }
  };

  const handleValidationResponse = (response) => {
    if (socket) {
      socket.emit('validationResponse', { response });
      setValidationPrompt('');
    }
  };

  return (
    <Paper style={{ padding: 20 }}>
      <Typography variant="h4">HTN Planner Visualization</Typography>

      <TextField
        label="Initial State"
        fullWidth
        margin="normal"
        value={initialState}
        onChange={(e) => setInitialState(e.target.value)}
      />

      <TextField
        label="Goal"
        fullWidth
        margin="normal"
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
      />

      <TextField
        label="Capabilities"
        fullWidth
        margin="normal"
        value={capabilities}
        onChange={(e) => setCapabilities(e.target.value)}
      />

      <Button
        variant="contained"
        color="primary"
        onClick={handleStart}
        style={{ marginTop: 10 }}
      >
        Start
      </Button>

      {realTimeLogs.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <Typography variant="h5">Real-Time Logs:</Typography>
          <pre>
            {realTimeLogs.map((log, index) => (
              <div key={index}>{log}</div>
            ))}
          </pre>
        </div>
      )}

      {validationPrompt && (
        <div style={{ marginTop: 20 }}>
          <Typography variant="h5">Validation Required:</Typography>
          <Typography>{validationPrompt}</Typography>
          <Button
            variant="contained"
            color="primary"
            onClick={() => handleValidationResponse('yes')}
            style={{ marginRight: 10 }}
          >
            Yes
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={() => handleValidationResponse('no')}
          >
            No
          </Button>
        </div>
      )}

      {finalPlan && (
        <div style={{ marginTop: 20 }}>
          <Typography variant="h5">Final Plan:</Typography>
          <pre>{JSON.stringify(finalPlan, null, 2)}</pre>
        </div>
      )}
    </Paper>
  );
}

export default HTNPlanner;
