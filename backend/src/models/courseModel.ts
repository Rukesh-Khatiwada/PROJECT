import mongoose from "mongoose";

const CourseSchema = new mongoose.Schema({
  name: { type: String, required: true },
  courseCode: { type: String, required: true, unique: true },
});

export const Course = mongoose.model("Course", CourseSchema);
