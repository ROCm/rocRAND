// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false)
{
    project.paths.construct_build_prefix()
    def command = """#!/usr/bin/env bash
                set -x
				# Command to create cppcheck result xml file
                cppcheck  --enable=all --inline-suppr --inconclusive --xml --xml-version=2 --output-file=cppcheck_result.xml ${project.paths.project_build_prefix}
                #Command to generate HTML report From the xml file
				        mkdir ./reports
				        cppcheck-htmlreport --title="cppcheck_Report" --report-dir=./reports --source-dir= ${project.paths.project_build_prefix} --file=cppcheck_result.xml
				"""

    platform.runCommand(this, command)
}

return this
