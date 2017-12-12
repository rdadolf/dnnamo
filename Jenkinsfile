def pipeline() {
  stage 'checkout'
    checkout scm
  stage 'setup_environment'
    sh 'build/build-docker-container.sh'
    sh 'build/check-data.sh'
    sh 'build/check-tensorflow.sh'
  stage 'static_checks'
    sh 'build/run-linter.sh'
    sh 'build/build-docs.sh'
  stage 'correctness'
    sh 'build/run-nosetests.sh'
    // docker tests
    // native tests
  stage 'performance'
    // container overhead
  stage 'reports'
    // docker plots & stats
    // native plots & stats
}

node('slave' && 'cpu') {
  pipeline()
}
