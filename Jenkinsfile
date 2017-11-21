pipeline {
    agent {
        docker { image 'python:3.6.3' }
    }
    stages {
        stage('Install') {
            steps {
                sh 'pip install -e .'
            }
        }
        stage('Test') {
            steps {
                sh 'python setup.py nosetests --with-xunit --xunit-file=test_results.xml'
            }
            post {
                always {
                    junit 'test_results.xml'
                }
            }
        }
        stage('Sanity - check') {
            steps {
                input 'Does everything look OK to release?'
            }
        }
    }
}