import shutil
import os
class DeployerEngine:
    def __init__(self, source):
        self.source = source
        self.root = self.source + "../../"
        self.output_path =self.root + "deployed_models/"
        

    def _copy_folder_contents(self, source_folder, destination_folder):
        try:
            shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
            print(f"Contents of '{source_folder}' copied successfully to '{destination_folder}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
    def _deploy_model(self):
        self._copy_folder_contents(self.source, self.output_path)
        self._copy_folder_contents(self.root + "model_files", self.output_path)
    def run(self):
        answer = input("Do you want to deploy models? (y/n)")
        if answer.lower() == "y":
            current_models = input("Do you want to deploy the current models? (y/n)")
            if current_models.lower() == "y":
                if os.path.exists(self.output_path):
                    shutil.rmtree(self.output_path)
                self._deploy_model()
                print("Model deployed!")
            else:
                self.source = "results/" + input("Enter the path of the model you want to deploy(example: 2024_05_09_11_08_56): ") + "/"
                self.__init__(self.source)
                if os.path.exists(self.output_path):
                    shutil.rmtree(self.output_path)
                self._deploy_model()
                print("Model deployed!")
        else:
            print("Model is not getting deployed!")