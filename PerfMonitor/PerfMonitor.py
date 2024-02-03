from jinja2 import Environment, FileSystemLoader
import os

class PerformaceMonitor:
    def __init__(self, path, report_template_dir= "PerfMonitor/Templates/"):
        self.path = path
        self.preprocessing_profiler  = None
        self.postprocessing_profiler = None
        self.inference_profiler      = None
        self.metrics_profiler        = None
        self.cpu_prof = []
        self.report_template_dir = os.path.join(os.getcwd(), report_template_dir)
        print(self.report_template_dir)

    def accumulate_cpu_runtime(self, new_record):
        self.cpu_prof.append(new_record)

    def add_profilers(self, preprocessing_profiler, postprocessing_profiler, inference_profiler, metrics_profiler):
        self.preprocessing_profiler = preprocessing_profiler
        self.postprocessing_profiler = postprocessing_profiler
        self.inference_profiler = inference_profiler        
        self.metrics_profiler = metrics_profiler

    def generate_reports(self):
        env = Environment(loader=FileSystemLoader(self.report_template_dir))
        template = env.get_template('report_template.j2')
        output_html = template.render(preprocessing=self.preprocessing_profiler, postprocessing=self.postprocessing_profiler, inference=self.inference_profiler, metrics=self.metrics_profiler)
        os.makedirs(f"{self.path}/perf_reports/", exist_ok=True)
        
        with open(f"{self.path}/perf_reports/report.html", 'w') as file:
            file.write(output_html)

        for i, prof in enumerate(self.cpu_prof):
            filename_memory = f"{self.path}/perf_reports/memory_usage_batch_{i+1}.txt"  # Create filename with index
            filename_runtime = f"{self.path}/perf_reports/runtime_batch_{i+1}.txt"  # Create filename with index
            with open(filename_memory, "w") as file:
                file.write(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
            with open(filename_runtime, "w") as file:
                file.write(prof.key_averages().table(sort_by="cuda_time_total"))


