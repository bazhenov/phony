use ndarray::{Array, Array2};
use std::convert::TryFrom;
use std::error::Error;
use std::path::Path;
use tensorflow as tf;
use tf::{Graph, Operation, Session, SessionOptions, SessionRunArgs, Status, Tensor, TensorType};

/// Микрофреймворк для решения задач при помощи библиотеки Tensorflow.
///
/// Подразумевается, что используя framework программист определяет следующие аспекты поведения:
/// * логику формирования контекста вычислений (`new_context`). Контекст позволяет выполнить любую требуемую
/// конвертацию или предобработку для примера. Так же контекст бывает удобен чтобы сохранить информацию о примере,
/// требуемую для полноценной интепретации результатов вычислений.
/// * как из tensorflow-модели достать точки входа и выхода (константы `GRAPH_INPUT_NAME` и `GRAPH_OUTPUT_NAME`).
/// Вход – это значение placeholder'а входного тензора.
/// Выход – это ответ системы, который содержит пометку класса или любую другую информацию, которая
/// является целью вычислений. И вход и выход являются именами соответствующх placeholder'ов/слоев в tensorflow-модели.
/// Поэтому, они должны быть определена в графе и в данной абстракции согласованным образом;
/// * логика получения тензора из примера (`tensors_from_example`). Каждый входящий пример (тип `Input`)
/// должен быть преобразован в тензор, размерность которого согласуется с размерностью входного placeholder'а
/// вычислительного графа tensorflow;
/// * логика преобразования выходного тензора в результат вычислений (`output_from_tensors`). Этот код интерпретирует
/// результат вычислений tensorflow и формирует ответ (тип `Output` – класс, пометка, координаты найденой
/// области и т.д.);
pub trait TensorflowProblem {
    /// Тип тензора-входа (`u32`/`f32` и т.д.)
    type TensorInputType: TensorType + Copy;

    /// Тип тензора-выхода (`u32`/`f32` и т.д.)
    type TensorOutputType: TensorType + Copy;

    /// Тип обрабатываемого примера. Например для задач классификации текстов: входной пример будет иметь тип `String`.
    type Input: ?Sized;

    /// Тип результата обработки. Для задач классификации это может быть `enum` с возможными классами.
    type Output;

    /// Имя placeholder'а входа в вычислительном графе tensorflow
    const GRAPH_INPUT_NAME: &'static str;

    /// Имя выходного слоя в вычислительном графе tensorflow
    const GRAPH_OUTPUT_NAME: &'static str;

    /// Создает контекст из входного примера.
    ///
    /// Контекст – абстракция позволяющая решить две задачи:
    /// * определить логику конвертации и предварительной обработки примера в вид более удобный для конвертации
    /// в тензор. Это позволяет упростить реализацию метода `tensors_from_example`. Это бывает полезно когда
    /// из одного примера необходимо генерировать несколько тензоров. Вынос предварительных вычислений
    /// в контекст позволяет избежать дублирующих вычислений;
    /// * сохранить информацию необходимую для интерпретации ответа вычислительного графа. Контекст доступен как
    /// в методу `tensors_from_example` так и методу `output_from_tensors`. Поэтому, его удобно использовать
    /// когда ответ вычислительного графа не является самостоятельгым и требует дальнейшей интерпретации на основании
    /// входных данных.
    fn new_context(example: &Self::Input) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;

    /// Формирует из примера тензор, который в последствии будет играть роль входных данных для tensorflow-графа.
    ///
    /// Тензор возвращаемый из этого метода по своей форме должен быть совместим с placeholder'ом вычислительного
    /// графа указанным в константе `GRAPH_INPUT_NAME`.
    fn tensors_from_example(&self, example: &Self::Input) -> Array2<Self::TensorInputType>;

    /// Формирует ответ системы на основании вычислений tensorflow.
    ///
    /// Принимает исходный пример, а также тензор из слоя указанного в `GRAPH_OUTPUT_NAME`. На основании этой
    /// информации формирует конечный ответ на задачу целиком.
    fn output_from_tensors(
        &self,
        example: &Self::Input,
        tensor: Array2<Self::TensorOutputType>,
    ) -> Self::Output;

    fn retrieve_input_output(&self, graph: &Graph) -> tf::Result<(Operation, Operation)> {
        let input = graph.operation_by_name_required(Self::GRAPH_INPUT_NAME)?;
        let output = graph.operation_by_name_required(Self::GRAPH_OUTPUT_NAME)?;

        Ok((input, output))
    }

    fn feed(
        &self,
        session: &Session,
        input_op: &Operation,
        output_op: &Operation,
        input: &Tensor<Self::TensorInputType>,
    ) -> tf::Result<Tensor<Self::TensorOutputType>> {
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&input_op, 0, input);
        let output_token = run_args.request_fetch(&output_op, 0);

        session.run(&mut run_args)?;
        run_args.fetch(output_token)
    }
}

fn tensor_from_ndarray<T, S>(input: Array<T, S>) -> Tensor<T>
where
    T: TensorType,
    S: ndarray::Dimension,
{
    let shape = input
        .shape()
        .iter()
        .cloned()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .expect("Unable to get ndarray::Array shape");

    let data = input.as_slice().expect("Can't get slice from ndarray");
    Tensor::new(&shape[..])
        .with_values(data)
        .expect("Can't build tensor")
}

fn ndarray_from_tensor<T: TensorType + Copy>(input: Tensor<T>) -> Array2<T> {
    let [rows, columns] = match *input.dims() {
        [a, b] => [a, b],
        _ => panic!("Should be 2-dimensional"),
    };
    let dims = [rows as usize, columns as usize];
    Array::from_iter(input.iter().cloned())
        .into_shape(dims)
        .expect("Unable to reshape")
}

pub struct TensorflowRunner {
    session: Session,
    graph: Graph,
}

impl TensorflowRunner {
    pub fn create_session<M: AsRef<Path>>(model_path: M) -> Result<Self, Status> {
        let mut graph = Graph::new();
        let tags: Vec<&str> = vec!["serve"];
        let session_options = SessionOptions::new();
        let session = Session::from_saved_model(&session_options, tags, &mut graph, model_path)?;
        Ok(TensorflowRunner { session, graph })
    }

    pub fn run_problem<P: TensorflowProblem>(
        &self,
        example: &P::Input,
    ) -> Result<P::Output, Box<dyn Error>> {
        let problem = P::new_context(&example)?;
        let (input_op, output_op) = problem.retrieve_input_output(&self.graph)?;

        let inputs = problem.tensors_from_example(&example);
        assert!(inputs.is_standard_layout(), "ndarray should be in standard (row-major) layout. Make sure you doesn't use ShapeBuilder::f() method when creating tensors");

        let tensor = tensor_from_ndarray(inputs);
        let output = problem
            .feed(&self.session, &input_op, &output_op, &tensor)
            .map(ndarray_from_tensor)
            .map(|output| problem.output_from_tensors(&example, output))?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::arr2;

    #[test]
    fn create_ndarray_from_tensor() {
        let tensor = Tensor::new(&[2, 2]).with_values(&[1, 2, 3, 4]);
        let ndarray = tensor.map(ndarray_from_tensor).unwrap();
        assert_eq!(ndarray.dim(), (2usize, 2usize));
        // Array values are expected in row-major format
        assert_eq!(ndarray.as_slice(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn create_tensor_from_ndarray() {
        let tensor = tensor_from_ndarray(arr2(&[[1, 2], [3, 4]]));
        assert_eq!(tensor.dims(), &[2, 2]);
        let content = tensor.iter().cloned().collect::<Vec<_>>();
        assert_eq!(content, vec![1, 2, 3, 4]);
    }
}
