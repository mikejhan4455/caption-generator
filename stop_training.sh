#!/bin/sh

kill $(pgrep python)
kill $(pgrep tensorboard)