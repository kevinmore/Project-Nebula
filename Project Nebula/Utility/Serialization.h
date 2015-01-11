#pragma once
#include <QtCore>
#include <Scene/GameObject.h>
#include <Scene/Camera.h>
#include <Scene/Managers/ModelManager.h>

/************************************************************************/
/*                           IO Streams                                 */
/************************************************************************/

/*
* Game Object
*/
// Order: Position -> Rotation -> Scaling
QDataStream& operator << (QDataStream& out, GameObject& object)
{
	out << object.position() << object.rotation() << object.scale();
	return out;
}

QDataStream& operator >> (QDataStream& in, GameObject& object)
{
	vec3 pos, rot, scale;
	in >> pos >> rot >> scale;

	object.setPosition(pos);
	object.setRotation(rot);
	object.setScale(scale);

	return in;
}

QDataStream& operator << (QDataStream& out, GameObject* object)
{
	out << object->position() << object->rotation() << object->scale();
	return out;
}

QDataStream& operator >> (QDataStream& in, GameObject* object)
{
	vec3 pos, rot, scale;
	in >> pos >> rot >> scale;

	object->setPosition(pos);
	object->setRotation(rot);
	object->setScale(scale);

	return in;
}

/*
* Model Manager
*/
// Order: vector size -> each pair(filename, gameobject)
QDataStream& operator << (QDataStream& out, QSharedPointer<ModelManager> object)
{
	int size = object->m_modelsInfo.size();
	out << size;

	for (int i = 0; i < size; ++i)
	{
		qDebug() << "Out Stream" << object->m_modelsInfo[i].first << object->m_modelsInfo[i].second->rotation();
		out << object->m_modelsInfo[i].first << object->m_modelsInfo[i].second;
	}
	
	return out;
}

QDataStream& operator >> (QDataStream& in, QSharedPointer<ModelManager> object)
{
	QVector<QPair<QString, GameObject*>> modelsInfoVector;

	int size;
	in >> size;

	QString fileName;
	GameObject* go = new GameObject;

	object->m_modelsInfo.clear();

	for (int i = 0; i < size; ++i)
	{
		in >> fileName >> go;
		qDebug() << "In Stream" << fileName << go->rotation();

		object->m_modelsInfo.push_back(qMakePair(fileName, go));
	}
	
	return in;
}