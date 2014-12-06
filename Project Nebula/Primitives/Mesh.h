#pragma once
#include <QString>

class Mesh
{
public:
	Mesh(const QString& name, unsigned int numIndices, unsigned int baseVertex, unsigned int baseIndex);

	virtual ~Mesh();

	void setName(const QString& name) { m_name = name; }
	QString name() const { return m_name; }

	unsigned int getNumIndices() const { return m_numIndices; }
	unsigned int getBaseVertex() const { return m_baseVertex; }
	unsigned int getBaseIndex()  const { return m_baseIndex;  }

private:
	QString m_name;

	unsigned int m_numIndices;
	unsigned int m_baseVertex;
	unsigned int m_baseIndex;
};

