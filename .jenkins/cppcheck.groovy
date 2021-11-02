#!/usr/bin/env groovy
@Library('rocJenkins@pong') _
import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false)
{
	project.paths.construct_build_prefix()
	def command = """
	#!/usr/bin/env bash
	set -x
    cppcheck --enable=all --inline-suppr --inconclusive --xml --xml-version=2 --output-file=cppcheck_result.xml ${project.paths.project_build_prefix}
    mkdir ./reports
    cppcheck-htmlreport --title="cppcheck_Report" --report-dir=./reports --source-dir= ${project.paths.project_build_prefix} --file=cppcheck_result.xml
    """
    platform.runCommand(this, command)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900'],centos7:['gfx906'],centos8:['gfx906'],sles15sp1:['gfx908']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each 
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCompileCommand(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCompileCommand([ubuntu16:['gfx906']], urlJobName)
        }
    }
}
