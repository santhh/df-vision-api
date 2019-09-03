package com.google.solutions.ml.api.vision;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.beam.runners.dataflow.options.DataflowPipelineOptions;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.PipelineResult;
import org.apache.beam.sdk.io.Compression;
import org.apache.beam.sdk.io.FileIO;
import org.apache.beam.sdk.io.FileIO.ReadableFile;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.gcp.bigquery.DynamicDestinations;
import org.apache.beam.sdk.io.gcp.bigquery.InsertRetryPolicy;
import org.apache.beam.sdk.io.gcp.bigquery.TableDestination;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.options.ValueProvider;
import org.apache.beam.sdk.options.ValueProvider.NestedValueProvider;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.GroupByKey;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.View;
import org.apache.beam.sdk.transforms.Watch;
import org.apache.beam.sdk.transforms.windowing.AfterProcessingTime;
import org.apache.beam.sdk.transforms.windowing.AfterWatermark;
import org.apache.beam.sdk.transforms.windowing.FixedWindows;
import org.apache.beam.sdk.transforms.windowing.Window;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.PCollectionView;
import org.apache.beam.sdk.values.ValueInSingleWindow;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.Instant;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.api.client.json.GenericJson;
import com.google.api.services.bigquery.model.TableCell;
import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import com.google.cloud.vision.v1.AnnotateImageRequest;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.cloud.vision.v1.BatchAnnotateImagesResponse;
import com.google.cloud.vision.v1.Feature;
import com.google.cloud.vision.v1.Feature.Builder;
import com.google.cloud.vision.v1.Image;
import com.google.cloud.vision.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.ImageSource;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;

public class VisionTextToBigQueryStreaming {
	public static final Logger LOG = LoggerFactory.getLogger(VisionTextToBigQueryStreaming.class);
	/** Default batch size if value not provided in execution. */
	private static final Integer DEFAULT_BATCH_SIZE = 2000;
	/** Default window interval to create side inputs for header records. */
	private static final Duration WINDOW_INTERVAL = Duration.standardSeconds(5);
	/** Default interval for polling files in GCS. */
	private static final Duration DEFAULT_POLL_INTERVAL = Duration.standardSeconds(30);

	private static final String BQ_TABLE_NAME = String.valueOf("VISION_API_FINDINGS");

	private static final DateTimeFormatter TIMESTAMP_FORMATTER = DateTimeFormat
			.forPattern("yyyy-MM-dd HH:mm:ss.SSSSSS");

	private static enum ALLOWED_FILE_EXTENSION {
		JPEG, JPG, PNG, GIF, BMP, WEBP, RAW, ICO, TIFF
	};

	public static void main(String[] args) {

		VisionApiPipelineOptions options = PipelineOptionsFactory.fromArgs(args).withValidation()
				.as(VisionApiPipelineOptions.class);
		run(options);

	}

	public static PipelineResult run(VisionApiPipelineOptions options) {

		Pipeline p = Pipeline.create(options);

		PCollection<KV<String, Iterable<String>>> imageFiles = p
				.apply("Poll Input Files",
						FileIO.match().filepattern(options.getInputFilePattern()).continuously(DEFAULT_POLL_INTERVAL,
								Watch.Growth.never()))
				.apply("Find Pattern Match", FileIO.readMatches().withCompression(Compression.AUTO))
				.apply("Get File Path", ParDo.of(new MapImageFiles()))
				.apply("Fixed Window(30 Sec)",
						Window.<KV<String, String>>into(FixedWindows.of(WINDOW_INTERVAL))
								.triggering(AfterWatermark.pastEndOfWindow().withEarlyFirings(
										AfterProcessingTime.pastFirstElementInPane().plusDelayOf(Duration.ZERO)))
								.discardingFiredPanes().withAllowedLateness(Duration.ZERO))
				.apply("Group By Bucket Name", GroupByKey.create());

		final PCollectionView<List<Feature>> featureList = imageFiles
				.apply("Create Feature List", ParDo.of(new CreateFeatureList(options.getFeatureType())))
				.apply(View.asList());

		imageFiles.apply("Process Images",
				ParDo.of(new ProcessImages(NestedValueProvider.of(options.getBatchSize(), batchSize -> {
					if (batchSize != null) {
						return batchSize;
					} else {
						return DEFAULT_BATCH_SIZE;
					}
				}), featureList)).withSideInputs(featureList))
				.apply("ConvertToTableRow", ParDo.of(new ConvertToTableRow())).apply("BQ Write",
						BigQueryIO.<KV<String, TableRow>>write()
								.to(new BQDestination(options.getDatasetName(), options.getVisionApiProjectId()))
								.withFormatFunction(element -> {
									return element.getValue();
								}).withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND)
								.withoutValidation()
								.withFailedInsertRetryPolicy(InsertRetryPolicy.retryTransientErrors())
								.withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED));

		return p.run();
	}

	public interface VisionApiPipelineOptions extends DataflowPipelineOptions {

		@Description("The file pattern to read records from (e.g. gs://bucket/file-*.jpg)")
		ValueProvider<String> getInputFilePattern();

		void setInputFilePattern(ValueProvider<String> value);

		@Description("List of features to be included seperated by comma(e.g. FACE_DETECTION,LANDMARK_DETECTION ")
		ValueProvider<String> getFeatureType();

		void setFeatureType(ValueProvider<String> value);

		@Description("Number of image files par API call. API accepts upto 2000 image files)")

		ValueProvider<Integer> getBatchSize();

		void setBatchSize(ValueProvider<Integer> value);

		@Description("Big Query data set must exist before the pipeline runs (e.g. pii-dataset")
		ValueProvider<String> getDatasetName();

		void setDatasetName(ValueProvider<String> value);

		@Description("Project id to be used for DLP Tokenization")
		ValueProvider<String> getVisionApiProjectId();

		void setVisionApiProjectId(ValueProvider<String> value);
	}

	private static class MapImageFiles extends DoFn<ReadableFile, KV<String, String>> {

		@ProcessElement
		public void processElement(ProcessContext c) {
			ReadableFile file = c.element();
			String imageFileName = file.getMetadata().resourceId().getFilename().toString();
			String bucketName = file.getMetadata().resourceId().getCurrentDirectory().toString();
			/** checking if it's a supported image format */
			String[] imageExtension = imageFileName.split("\\.", 2);
			if (imageExtension.length == 2) {
				for (ALLOWED_FILE_EXTENSION ext : ALLOWED_FILE_EXTENSION.values()) {

					if (ext.name().equalsIgnoreCase(imageExtension[1].trim())) {

						c.output(KV.of(bucketName, imageFileName));

					}
				}
			}
		}
	}

	private static class CreateFeatureList extends DoFn<KV<String, Iterable<String>>, Feature> {

		private ValueProvider<String> featureConfig;
		private Gson json;

		public CreateFeatureList(ValueProvider<String> featureConfig) {
			this.featureConfig = featureConfig;
		}

		@Setup
		public void setup() {
			json = new Gson();
		}

		@ProcessElement
		public void processElement(ProcessContext c) throws InvalidProtocolBufferException {

			List<GenericJson> features = json.fromJson(featureConfig.get(), new TypeToken<List<GenericJson>>() {
			}.getType());
			features.forEach(feature -> {
				Builder featureBuilder = Feature.newBuilder();
				feature.entrySet().forEach(set -> {
					FieldDescriptor key = Builder.getDescriptor().findFieldByName(set.getKey().toString().trim());
					Object value = set.getValue();

					if (key != null && value != null) {
						if (key.getJsonName().equalsIgnoreCase("type")) {
							featureBuilder.setType(Feature.Type.valueOf(value.toString().trim()));
						} else if (key.getJsonName().equalsIgnoreCase("maxResults")) {
							featureBuilder.setMaxResults(Integer.parseInt(value.toString().trim()));
						} else {
							featureBuilder.setField(key, value);
						}
					} else {

						LOG.error("Can't find the field for {}", set.getKey().toString());
					}
				});

				c.output(featureBuilder.build());

			});

		}
	}

	private static class ConvertToTableRow extends DoFn<KV<String, AnnotateImageResponse>, KV<String, TableRow>> {

		Gson gson;

		@Setup
		public void setup() {
			gson = new Gson();
		}

		@ProcessElement
		public void processElement(ProcessContext c) throws InvalidProtocolBufferException {
			AnnotateImageResponse imageResponse = c.element().getValue();
			String imageFileName = c.element().getKey();
			String timestamp = TIMESTAMP_FORMATTER.print(Instant.now().toDateTime(DateTimeZone.UTC));
			
			GenericJson genericJson = convertToValidJson(imageResponse);
			
			genericJson.entrySet().forEach(element->{
				List<TableCell> cells = new ArrayList<>();
				TableRow row = new TableRow();
				cells.add(new TableCell().set("file_name", imageFileName));
				row.set("file_name", imageFileName);
				cells.add(new TableCell().set("transaction_timestamp", timestamp));
				row.set("transaction_timestamp", timestamp);

				cells.add(new TableCell().set("feature_type", element.getKey()));
				row.set("feature_type", element.getKey());

				cells.add(new TableCell().set("raw_json_response", element.getValue().toString()));
				row.set("raw_json_response", gson.toJson(element.getValue()));
				row.setF(cells);
				//LOG.info("Row {} ", row.toString());
				c.output(KV.of(BQ_TABLE_NAME, row));
			});
			
			
			
		}
		
		public GenericJson convertToValidJson(AnnotateImageResponse response) throws InvalidProtocolBufferException {
			
			
			return gson.fromJson(JsonFormat.printer().print(response), 
					new TypeToken<GenericJson>(){}.getType());
			
		}
	}

	private static class ProcessImages extends DoFn<KV<String, Iterable<String>>, KV<String, AnnotateImageResponse>> {

		private ValueProvider<Integer> batchSize;
		private PCollectionView<List<Feature>> featureList;
		private ImageAnnotatorClient visionApiClient;
		private List<AnnotateImageRequest> requests;

		public ProcessImages(ValueProvider<Integer> batchSize, PCollectionView<List<Feature>> featureList) {

			this.batchSize = batchSize;
			this.featureList = featureList;
			this.requests = new ArrayList<>();
		}

		@StartBundle
		public void startBundle() {
			try {
				visionApiClient = ImageAnnotatorClient.create();
			} catch (IOException e) {
				LOG.error("Failed to create Vision API Service Client", e.getMessage());
				throw new RuntimeException(e);
			}

		}

		@FinishBundle
		public void finishBundle() {
			if (visionApiClient != null) {
				visionApiClient.close();
			}
		}

		@ProcessElement
		public void processElement(ProcessContext c) {

			Iterator<String> imgItr = c.element().getValue().iterator();
			AtomicInteger index = new AtomicInteger(0);
			// long count = StreamSupport.stream(c.element().getValue().spliterator(),
			// false).count();
			// LOG.info("Count {}",count);
			String bucketName = c.element().getKey();
			List<Feature> features = c.sideInput(featureList);
			while (imgItr.hasNext()) {
				String imagePath = String.format("%s%s", bucketName, imgItr.next());
				Image image = Image.newBuilder().setSource(ImageSource.newBuilder().setImageUri(imagePath).build())
						.build();
				requests.add(AnnotateImageRequest.newBuilder().setImage(image).addAllFeatures(features).build());

			}

			BatchAnnotateImagesResponse response = visionApiClient.batchAnnotateImages(requests);
			List<AnnotateImageResponse> responses = response.getResponsesList();

			responses.forEach(imageResponse -> {

				c.output(KV.of(requests.get(index.getAndIncrement()).getImage().getSource().getImageUri(), imageResponse));

			});

		}
	}

	public static class BQDestination extends DynamicDestinations<KV<String, TableRow>, KV<String, TableRow>> {

		private ValueProvider<String> datasetName;
		private ValueProvider<String> projectId;

		public BQDestination(ValueProvider<String> datasetName, ValueProvider<String> projectId) {
			this.datasetName = datasetName;
			this.projectId = projectId;
		}

		@Override
		public KV<String, TableRow> getDestination(ValueInSingleWindow<KV<String, TableRow>> element) {
			String key = element.getValue().getKey();
			String tableName = String.format("%s:%s.%s", projectId.get(), datasetName.get(), key);
			LOG.debug("Table Name {}", tableName);
			return KV.of(tableName, element.getValue().getValue());
		}

		@Override
		public TableDestination getTable(KV<String, TableRow> destination) {
			TableDestination dest = new TableDestination(destination.getKey(),
					"vision api data from dataflow");
			LOG.debug("Table Destination {}", dest.getTableSpec());
			return dest;
		}

		@Override
		public TableSchema getSchema(KV<String, TableRow> destination) {

			TableRow bqRow = destination.getValue();
			TableSchema schema = new TableSchema();
			List<TableFieldSchema> fields = new ArrayList<TableFieldSchema>();
			List<TableCell> cells = bqRow.getF();
			for (int i = 0; i < cells.size(); i++) {
				Map<String, Object> object = cells.get(i);
				String header = object.keySet().iterator().next();
				/** currently all BQ data types are set to String */
				if(!header.equals("raw_json_response"))
					fields.add(new TableFieldSchema().setName(header).setType("STRING"));
				else {
					List<TableFieldSchema> nestedSchema = new ArrayList<>();
					nestedSchema.add(new TableFieldSchema().setName("json_value").setType("STRING"));
					fields.add(new TableFieldSchema().setName(header).setType("RECORD").setFields(nestedSchema));

				}
			}

			schema.setFields(fields);
			return schema;
		}
	}

}
