pipeline{
	agent any
	stages{
		stage('build'){
			steps{
				echo 'executing shell'
				echo '1'
			}
		}
	
	}
	
	post{
		always{
			echo 'pipeline successfully'
			publishHTML(target:[allowMissing: false,
			alwaysLinkToLastBuild:true,
			keepAll: true,
			reportDir: 'reports',
			reportFiles:'*.html',
			reportName: 'My reports',
			reportTitles: 'The Report'])
		}
	
	}


}