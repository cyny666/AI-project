

- 接口类文件为aigcmn.py。

- 调用步骤与代码演示：

  1. `test=AiGcMn("generator_params.pth","discriminator_params.pth")`

     创建AiGcMn接口类的实例，需要传入2个str类型的参数(是预先训练好并保存的模型参数)，分别为包含文件中的2个文件的文件名：`"generator_params.pth"`和`"discriminator_params.pth"`

  2. `test.misc()`

     `misc()`为其他处理函数，用于加载模型参数和其他中间过程的处理

  3. ` output_tensor=test.generate(target=input)`

     接口函数为generate：有一个target参数，类型为一个整型的n维torch.Tensor（n是batch的大小，每个整数在0~9范围内，代表需要生成的数字）；

     该函数的返回`output_tensor`为n * 1 * 28 * 28的tensor（n是batch的大小，每个1 * 28 * 28的tensor表示随机生成的数字图像）；

     并且会在当前目录生成一个1行n列的对应数字的网格图像

- 示例代码：

  ```python
  if __name__=='__main__':
  
      data = [0,1,2,3,4,5,6,7,8,9]
  
      # input为输入的整数型n维tensor
      input=torch.tensor(data)
  
      # 创建AiGcMn接口类的实例，需要传入2个参数(是预先训练好的模型参数)
      test=AiGcMn("generator_params.pth","discriminator_params.pth")
      # misc()为其他处理函数，用于加载和中间过程的处理
      test.misc()
      """
      接口函数generate:target参数为一个整型的n维torch.Tensor，
      （n是batch的大小，每个整数在0~9范围内，代表需要生成的数字）
      该函数返回n*1*28*28的tensor
      （n是batch的大小，每个1*28*28的tensor表示随机生成的数字图像）
      并且会在当前目录生成一行对应数字的网格图像
      """
      output_tensor=test.generate(target=input)
      
      print("输出图像tensor.size: ",output_tensor.size())
  ```

  终端输出为：`输出图像tensor.size:  torch.Size([10, 1, 28, 28])`


