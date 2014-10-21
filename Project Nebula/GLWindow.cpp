#include <gl/glew.h>
#include <GLWindow.h>
#include <QtGui/QDesktopWidget>
#include <QtCore/QDebug>
#include <Utility/MeshGenerator.h>



GLWindow::GLWindow(QWidget *parent) : QGLWidget(QGLFormat(/* Additional format options */), parent)
{
	//	setFormat(QGLFormat(QGL::Rgba | QGL::DoubleBuffer | QGL::DepthBuffer));
	//	m_allRenderableObjects.reserve(1);
	m_frameCount = 0;
	handModel = NULL;
	hand = NULL;
	lightAngle = 0;
	handRotationAngle = 1;
	handRotationDirection = 1;

	m_timer = new QTimer(this);
	connect(m_timer, SIGNAL(timeout()), this, SLOT(updateLoop()));
	m_timer->start(0);
	m_elaTimer = new QElapsedTimer();
	m_elaTimer->start();

	m_importer = new MeshImporter();
}

QSize GLWindow::sizeHint() const
{
	return QSize(1024, 768);
}

GLWindow::~GLWindow(void)
{
	//	if(hand) hand->freeSkeleton(hand);
	//	glUseProgram(0);
	// 	Shader* temp = 0;
	// 	for (unsigned int i = 0; i < m_runningShaderPrograms.size(); ++i)
	// 	{
	// 		temp = m_runningShaderPrograms[i];
	// 		temp->DelteBuffers();
	// 		glDeleteProgram(temp->ShaderProgramID());
	// 	}
}

void GLWindow::initializeGL()
{

#ifdef WIN32
	glActiveTexture = (PFNGLACTIVETEXTUREPROC) wglGetProcAddress((LPCSTR) "glActiveTexture");
#endif

	glShadeModel(GL_SMOOTH);
	glClearDepth( 1.0 );

	// enable the depth buffer
	glEnable(GL_DEPTH_TEST);
	//glDepthFunc( GL_LEQUAL );

	//glEnable(GL_CULL_FACE);

	qglClearColor(QColor(40, 40, 40));


	// spot light

	coloringShaderProgram.addShaderFromSourceFile(QGLShader::Vertex, "../Resource/Shaders/coloringVertexShader.vsh");
	coloringShaderProgram.addShaderFromSourceFile(QGLShader::Fragment, "../Resource/Shaders/coloringFragmentShader.fsh");
	coloringShaderProgram.link();

	QVector<QVector3D> spotlightVertices;
	QVector<QVector3D> spotlightColors;

	spotlightVertices << QVector3D(   0,    0.1,    0) << QVector3D(-0.05,    0,  0.05) << QVector3D( 0.05,    0,  0.05) // Front
		<< QVector3D(   0,    0.1,    0) << QVector3D( 0.05,    0, -0.05) << QVector3D(-0.05,    0, -0.05) // Back
		<< QVector3D(   0,    0.1,    0) << QVector3D(-0.05,    0, -0.05) << QVector3D(-0.05,    0,  0.05) // Left
		<< QVector3D(   0,    0.1,    0) << QVector3D( 0.05,    0,  0.05) << QVector3D( 0.05,    0, -0.05) // Right
		<< QVector3D(-0.05,    0, -0.05) << QVector3D( 0.05,    0, -0.05) << QVector3D( 0.05,    0,  0.05) // Bottom
		<< QVector3D( 0.05,    0,  0.05) << QVector3D(-0.05,    0,  0.05) << QVector3D(-0.05,    0, -0.05);
	spotlightColors << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Front
		<< QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Back
		<< QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Left
		<< QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) << QVector3D(0.1, 0.1, 0.1) // Right
		<< QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9) // Bottom
		<< QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9) << QVector3D(  0.9,   0.9,   0.9);

	numSpotlightVertices = 18;
	spotlightBuffer.create();
	spotlightBuffer.bind();
	spotlightBuffer.setUsagePattern(QGLBuffer::DynamicDraw);
	spotlightBuffer.allocate(numSpotlightVertices * (3 + 3) * sizeof(GLfloat));

	int offset = 0;
	spotlightBuffer.write(offset, spotlightVertices.constData(), numSpotlightVertices * 3 * sizeof(GLfloat));
	offset += numSpotlightVertices * 3 * sizeof(GLfloat);
	spotlightBuffer.write(offset, spotlightColors.constData(), numSpotlightVertices * 3 * sizeof(GLfloat));

	spotlightBuffer.release();
	/***********************************************************************************************************/

	lightingShaderProgram.addShaderFromSourceFile(QGLShader::Vertex, "../Resource/Shaders/lightingVertexShader.vsh");
	lightingShaderProgram.addShaderFromSourceFile(QGLShader::Fragment, "../Resource/Shaders/lightingFragmentShader.fsh");
	lightingShaderProgram.link();

	skinningShaderProgram.addShaderFromSourceFile(QGLShader::Vertex, "../Resource/Shaders/skinning.vs");
	skinningShaderProgram.addShaderFromSourceFile(QGLShader::Fragment, "../Resource/Shaders/skinning.fs");
	skinningShaderProgram.link();

	skinningShaderProgram2.addShaderFromSourceFile(QGLShader::Vertex, "../Resource/Shaders/skinningVertexShader.vsh");
	skinningShaderProgram2.addShaderFromSourceFile(QGLShader::Fragment, "../Resource/Shaders/skinningFragmentShader.fsh");
	skinningShaderProgram2.link();


	// make a hand
	hand = MeshGenerator::makeHand();

	// import the hand mesh
	if (m_importer->loadMeshFromFile("../Resource/Models/female/female.dae"))
	{
		qDebug() << "Model loaded successfully!";
		//		qDebug() << m_importer->m_pScene->HasAnimations();
		// 		handModel = m_importer->getSkeleton();
		// 		mat4 transform;
		// 		transform.scale(0.01, 0.01, 0.01);
		// 		Bone::sortSkeleton(handModel);
	}
}

void GLWindow::resizeGL( int w, int h )
{
	if ( h == 0 ) h = 1;

	pMatrix.setToIdentity();
	pMatrix.perspective(60.0, (float) w / (float) h, 0.001, 1000);
	// set the view port the the full qt window
	glViewport( 0, 0, w, h );

}

void GLWindow::paintGL()
{
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	/*********************set up the world to view matrix****************************************/
	vMatrix.setToIdentity();
	vMatrix.lookAt(m_camera.Position(), m_camera.ViewDirection() + m_camera.Position(), m_camera.Up());

	/*********************set up the spinning spot light****************************************/
	mMatrix.setToIdentity();

	QMatrix4x4 mvMatrix;
	mvMatrix = vMatrix * mMatrix;

	normalMatrix = mvMatrix.normalMatrix();

	QMatrix4x4 lightTransformation;
	lightTransformation.rotate(lightAngle, 0, 1, 0);

	lightPosition = lightTransformation * QVector3D(0, 0.14, 0.1);
	mMatrix.setToIdentity();
	mMatrix.translate(lightPosition);
	mMatrix.rotate(lightAngle, 0, 1, 0);
	mMatrix.rotate(45, 1, 0, 0);
	mMatrix.scale(0.1);

	coloringShaderProgram.bind();

	coloringShaderProgram.setUniformValue("mvpMatrix", pMatrix * vMatrix * mMatrix);

	spotlightBuffer.bind();
	int offset = 0;
	coloringShaderProgram.setAttributeBuffer("vertex", GL_FLOAT, offset, 3, 0);
	coloringShaderProgram.enableAttributeArray("vertex");
	offset += numSpotlightVertices * 3 * sizeof(GLfloat);
	coloringShaderProgram.setAttributeBuffer("color", GL_FLOAT, offset, 3, 0);
	coloringShaderProgram.enableAttributeArray("color");
	spotlightBuffer.release();

	glDrawArrays(GL_TRIANGLES, 0, numSpotlightVertices);

	coloringShaderProgram.disableAttributeArray("vertex");
	coloringShaderProgram.disableAttributeArray("color");
	coloringShaderProgram.release();

	/*********************Render other objects*************************************/

	// change the animation direction every 70 frames
	++m_frameCount;
	if (m_frameCount % 70 == 0)
	{
		handRotationAngle *= -1;
	}

	/** render the hand represented by cylinders **/
	mat4 transform;
	// rotate the palm around the Y axis, for demo purpose
	transform.rotate(0.2, vec3(0, 1, 0));
	hand->m_localTransform *= transform;
	Bone::sortSkeleton(hand);
	// only move the 5 fingers, not the palm
	// thumb
	transform.setToIdentity();
	transform.rotate(-handRotationAngle, vec3(1, 0, 0));
	Bone::configureSkeleton(hand->getChild(0), transform);
	// other 4 fingers
	transform.setToIdentity();
	transform.rotate(-1.5*handRotationAngle, vec3(1, 0, 0));
	for (int i = 1; i < hand->childCount(); ++i)
	{
		Bone::configureSkeleton(hand->getChild(i), transform);
	}
//	renderSkeleton(hand);

	
	/** render the imported hand model **/
	//if(!handModel) return; // the model may not be imported successfully
	// this render path has no animations
//  	transform.setToIdentity();
// 	for (int i = 0; i < m_importer->m_Meshes.size(); ++i)
// 	{
// 		renderMesh(lightingShaderProgram, *m_importer->m_Meshes[i], transform);
// 	}
 	
	// this render path has animations
	renderModel(m_importer);	
}

void GLWindow::renderMesh( QGLShaderProgram &shader, MeshData &mesh, mat4 &modelToWorldMatrix )
{
	// calculate MV Matrix
	mat4 mvMatrix = vMatrix * modelToWorldMatrix;

	// active the shader
	shader.bind();

	// set the uniform values
	shader.setUniformValue("mvpMatrix", pMatrix * mvMatrix);
	shader.setUniformValue("mvMatrix", mvMatrix);
	shader.setUniformValue("normalMatrix", normalMatrix);
	shader.setUniformValue("lightPosition", mvMatrix * lightPosition);
	shader.setUniformValue("ambientColor", QColor(32, 32, 32));
	shader.setUniformValue("diffuseColor", QColor(128, 128, 128));
	shader.setUniformValue("specularColor", QColor(255, 255, 255));
	shader.setUniformValue("ambientReflection", (GLfloat) 1.0);
	shader.setUniformValue("diffuseReflection", (GLfloat) 1.0);
	shader.setUniformValue("specularReflection", (GLfloat) 1.0);
	shader.setUniformValue("shininess", (GLfloat) 100.0);
	shader.setUniformValue("texture", 0);
	if (mesh.material->textureFile!=NULL)
	{
		shader.setUniformValue("useTexture", true);
		mesh.material->textureFile->bind(GL_TEXTURE0);
	} 
	else
	{
		shader.setUniformValue("useTexture", false);
	}

	// active the buffer from the mesh itself 
	// pass in values
	// lastly, release the buffer
	mesh.vertexBuff.bind();
	int offset = 0;
	shader.setAttributeBuffer("vertex", GL_FLOAT, 0, 3, sizeof(Vertex));
	shader.enableAttributeArray("vertex");
	offset += sizeof(mesh.vertices[0].postition);

	shader.setAttributeBuffer("color", GL_FLOAT, offset, 4, sizeof(Vertex));
	shader.enableAttributeArray("color");
	offset += sizeof(mesh.vertices[0].color);

	shader.setAttributeBuffer("normal", GL_FLOAT, offset, 3, sizeof(Vertex));
	shader.enableAttributeArray("normal");
	offset += sizeof(mesh.vertices[0].normal);

	if (mesh.material->textureFile!=NULL)
	{
		shader.setAttributeBuffer("textureCoordinate", GL_FLOAT, offset, 2, sizeof(Vertex));
		shader.enableAttributeArray("textureCoordinate");
	}
	mesh.vertexBuff.release();

	// draw the mesh here
	glDrawElements(GL_TRIANGLES, mesh.numIndices, GL_UNSIGNED_SHORT, mesh.indices);

	// clean up
	shader.disableAttributeArray("vertex");
	shader.disableAttributeArray("color");
	shader.disableAttributeArray("normal");
	if (mesh.material->textureFile!=NULL) 
	{
		mesh.material->textureFile->release();
		shader.disableAttributeArray("textureCoordinate");
	}

	shader.release();
}

void GLWindow::renderBone( QGLShaderProgram &shader, MeshData &mesh)
{

	// calculate MV Matrix
	mat4 mMatrix;
	mat4 mvMatrix = vMatrix * mMatrix;

	// active the shader
	shader.bind();

	// set the uniform values
	QVector<mat4> Transforms;
	m_importer->BoneTransform((float)m_elaTimer->elapsed()/1000, Transforms);
	for (int i = 0 ; i < Transforms.size() ; i++) 
	{
	//	glUniformMatrix4fv(m_boneLocation[i], 1, GL_TRUE, (const GLfloat*)transform.constData());
		shader.setUniformValue(m_boneLocation[i], Transforms[i].transposed());
	}
	shader.setUniformValue("gWVP", pMatrix * mvMatrix);
	shader.setUniformValue("gWorld", mvMatrix);
	shader.setUniformValue("gEyeWorldPos", m_camera.Position());
	
	shader.setUniformValue("gColorMap", 0);
	shader.setUniformValue("gDirectionalLight.Base.Color", vec3(1.0f, 1.0f, 1.0f));
	shader.setUniformValue("gDirectionalLight.Base.AmbientIntensity", 0.55f);
	shader.setUniformValue("gDirectionalLight.Base.DiffuseIntensity", 0.9f);
	shader.setUniformValue("gDirectionalLight.Direction", vec3(1.0f, -0.5f, -0.5f));
	shader.setUniformValue("gMatSpecularIntensity", 0);
	shader.setUniformValue("gSpecularPower", 0);
	shader.setUniformValue("gPointLights[0].Base.Color", vec3(1.0f, 1.0f, 1.0f));
	shader.setUniformValue("gPointLights[0].Base.AmbientIntensity", 0.55f);
	shader.setUniformValue("gPointLights[0].Base.DiffuseIntensity", 0.9f);
	shader.setUniformValue("gPointLights[0].Position", vec3(10, 10, 10));
	shader.setUniformValue("gPointLights[0].Atten", 0.8f);


	shader.setUniformValue("gNumSpotLights", 0);
	
	// active the buffer from the mesh itself 
	// pass in values
	// lastly, release the buffer
	mesh.vertexBuff.bind();
	int offset = 0;
	shader.setAttributeBuffer("Position", GL_FLOAT, 0, 3, sizeof(Vertex));
	shader.enableAttributeArray("Position");
	offset += sizeof(mesh.vertices[0].postition);

	shader.setAttributeBuffer("color", GL_FLOAT, offset, 4, sizeof(Vertex));
	shader.enableAttributeArray("color");
	offset += sizeof(mesh.vertices[0].color);

	shader.setAttributeBuffer("Normal", GL_FLOAT, offset, 3, sizeof(Vertex));
	shader.enableAttributeArray("Normal");
	offset += sizeof(mesh.vertices[0].normal);

	shader.setAttributeBuffer("TexCoord", GL_FLOAT, offset, 2, sizeof(Vertex));
	shader.enableAttributeArray("TexCoord");
	offset += sizeof(mesh.vertices[0].texCoord);

	shader.setAttributeBuffer("BoneIDs", GL_FLOAT, offset, 4, sizeof(Vertex));
	shader.enableAttributeArray("BoneIDs");
	offset += sizeof(mesh.vertices[0].boneIDs);

	shader.setAttributeBuffer("Weights", GL_FLOAT, offset, 4, sizeof(Vertex));
	shader.enableAttributeArray("Weights");
	offset += sizeof(mesh.vertices[0].boneWeights);

	mesh.vertexBuff.release();

	// draw the mesh here
	glDrawElements(GL_TRIANGLES, mesh.numIndices, GL_UNSIGNED_SHORT, mesh.indices);

	// clean up
	shader.disableAttributeArray("Position");
	shader.disableAttributeArray("color");
	shader.disableAttributeArray("Normal");
	shader.disableAttributeArray("TexCoord");
	shader.disableAttributeArray("BoneIDs");
	shader.disableAttributeArray("Weights");
	if (mesh.material->textureFile!=NULL) 
	{
		mesh.material->textureFile->release();
		shader.disableAttributeArray("TexCoord");
	}

	shader.release();
}

void GLWindow::renderBone2( QGLShaderProgram &shader, MeshData &mesh )
{
	// calculate MV Matrix
	mat4 modelToWorldMatrix;
	mat4 mvMatrix = vMatrix * modelToWorldMatrix;

	// active the shader
	shader.bind();

	// set the uniform values
	// set the uniform values
	QVector<mat4> Transforms;
	m_importer->BoneTransform((float)m_elaTimer->elapsed()/1000, Transforms);
	for (int i = 0 ; i < Transforms.size() ; i++) 
	{
		char Name[128];
		memset(Name, 0, sizeof(Name));
		_snprintf_s(Name, sizeof(Name), "gBones[%d]", i);
		shader.setUniformValue(Name, Transforms[i].transposed());
	}

	shader.setUniformValue("mvpMatrix", pMatrix * mvMatrix);
	shader.setUniformValue("mvMatrix", mvMatrix);
	shader.setUniformValue("normalMatrix", normalMatrix);
	shader.setUniformValue("lightPosition", mvMatrix * lightPosition);
	shader.setUniformValue("ambientColor", QColor(32, 32, 32));
	shader.setUniformValue("diffuseColor", QColor(128, 128, 128));
	shader.setUniformValue("specularColor", QColor(255, 255, 255));
	shader.setUniformValue("ambientReflection", (GLfloat) 1.0);
	shader.setUniformValue("diffuseReflection", (GLfloat) 1.0);
	shader.setUniformValue("specularReflection", (GLfloat) 1.0);
	shader.setUniformValue("shininess", (GLfloat) 100.0);
	shader.setUniformValue("texture", 0);
	if (mesh.material->textureFile!=NULL)
	{
		shader.setUniformValue("useTexture", true);
		mesh.material->textureFile->bind(GL_TEXTURE0);
	} 
	else
	{
		shader.setUniformValue("useTexture", false);
	}


	// active the buffer from the mesh itself 
	// pass in values
	// lastly, release the buffer
	mesh.vertexBuff.bind();
	int offset = 0;
	shader.setAttributeBuffer("vertex", GL_FLOAT, 0, 3, sizeof(Vertex));
	shader.enableAttributeArray("vertex");
	offset += sizeof(mesh.vertices[0].postition);

	shader.setAttributeBuffer("color", GL_FLOAT, offset, 4, sizeof(Vertex));
	shader.enableAttributeArray("color");
	offset += sizeof(mesh.vertices[0].color);

	shader.setAttributeBuffer("normal", GL_FLOAT, offset, 3, sizeof(Vertex));
	shader.enableAttributeArray("normal");
	offset += sizeof(mesh.vertices[0].normal);

	shader.setAttributeBuffer("textureCoordinate", GL_FLOAT, offset, 2, sizeof(Vertex));
	shader.enableAttributeArray("textureCoordinate");
	offset += sizeof(mesh.vertices[0].texCoord);

	shader.setAttributeBuffer("BoneIDs", GL_INT, offset, 4, sizeof(Vertex));
	shader.enableAttributeArray("BoneIDs");
	offset += sizeof(mesh.vertices[0].boneIDs);

	shader.setAttributeBuffer("Weights", GL_FLOAT, offset, 4, sizeof(Vertex));
	shader.enableAttributeArray("Weights");
	offset += sizeof(mesh.vertices[0].boneWeights);

	mesh.vertexBuff.release();

	// draw the mesh here
	glDrawElements(GL_TRIANGLES, mesh.numIndices, GL_UNSIGNED_SHORT, mesh.indices);

	// clean up
	shader.disableAttributeArray("vertex");
	shader.disableAttributeArray("color");
	shader.disableAttributeArray("normal");
	shader.disableAttributeArray("textureCoordinate");
	shader.disableAttributeArray("BoneIDs");
	shader.disableAttributeArray("Weights");
	if (mesh.material->textureFile!=NULL) 
	{
		mesh.material->textureFile->release();
		shader.disableAttributeArray("TexCoord");
	}

	shader.release();
}

void GLWindow::renderSkeleton( Bone* root )
{
	if(!root) return; // empty skeleton
	QVector<MeshData> meshes = root->getMeshData();
	for (int i = 0; i < meshes.size(); ++i)
	{
		renderMesh(lightingShaderProgram, meshes[i], root->m_globalTransform);
	}

	for (int i = 0; i < root->childCount(); ++i)
	{
		renderSkeleton(root->getChild(i));
	}

}


void GLWindow::renderModel( MeshImporter *importer )
{
	mat4 mMatrix;
	mMatrix.setToIdentity();
	QVector<MeshData*> meshes = importer->m_Meshes;
	for (int i = 0; i < meshes.size(); ++i)
	{
		renderBone2(skinningShaderProgram2, *meshes[i]);
	}
}

void GLWindow::keyPressEvent( QKeyEvent * e )
{
	switch (e->key())
	{
	case Qt::Key_F11:
		if ( isFullScreen() )
		{
			showNormal();
			resizeToScreenCenter();
		}
		else
		{
			showFullScreen();
		}
		updateGL();
		break;
	case Qt::Key_Escape:
		close();
	case Qt::Key_Plus:
		m_camera.movementSpeed += 0.5;
	case Qt::Key_Minus:
		m_camera.movementSpeed -= 0.5;
	}

}

void GLWindow::resizeToScreenCenter()
{
	setGeometry( 0, 0, 1024, 768 );
	move(QApplication::desktop()->screen()->rect().center() - rect().center());
}

void GLWindow::updateLoop()
{
	lightAngle += 0.5;
	while (lightAngle >= 360) {
		lightAngle -= 360;
	}

	checkKeyState();
	updateGL();

}

void GLWindow::mousePressEvent( QMouseEvent *e )
{
	m_camera.LastMousePosition(vec2(e->pos()));
	e->accept();
}

void GLWindow::mouseMoveEvent( QMouseEvent *e)
{
	if (e->buttons() == Qt::RightButton)
	{
		m_camera.lookAround(vec2(e->x(), e->y()));
		updateGL();
	}

	m_camera.LastMousePosition(vec2(e->pos()));

	e->accept();
}

void GLWindow::wheelEvent( QWheelEvent *e )
{
	int delta = e->delta();

	if (e->orientation() == Qt::Vertical) {
		if (delta < 0) {
			m_camera.moveBackward();
		} else if (delta > 0) {
			m_camera.moveForward();
		}

		updateGL();
	}

	e->accept();
}

void GLWindow::checkKeyState()
{
	if (GetAsyncKeyState('W')) m_camera.moveForward();
	if (GetAsyncKeyState('S')) m_camera.moveBackward();
	if (GetAsyncKeyState('A')) m_camera.moveLeft();
	if (GetAsyncKeyState('D')) m_camera.moveRight();
	if (GetAsyncKeyState('R')) m_camera.moveUp();
	if (GetAsyncKeyState('F')) m_camera.moveDown();
}
