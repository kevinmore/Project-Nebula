/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/LinearVertexCalculator/hkLinearVertexCalculator.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>

hkLinearVertexCalculator::hkLinearVertexCalculator()
{
	m_vertexBuffer = HK_NULL;
	m_isStarted = false;

	// Enable the epsilon
	disableAllElements();
	enableElement(hkVertexFormat::USAGE_TEX_COORD, hkReal(1e-4f));
	enableElement(hkVertexFormat::USAGE_NORMAL, hkReal(1e-3f));
}

void HK_CALL hkLinearVertexCalculator::calcBarycentricCoordinates(hkVector4Parameter pos, hkVector4Parameter t0, hkVector4Parameter t1, hkVector4Parameter t2, hkVector4& lambdas)
{
    hkVector4 R, Q;
    Q.setSub(t0, t1);      
    R.setSub(t2, t1);

    const hkSimdReal QQ = Q.lengthSquared<3>();
    const hkSimdReal RR = R.lengthSquared<3>();
    const hkSimdReal QR = R.dot<3>(Q);

    const hkSimdReal QQRR = QQ * RR;
    const hkSimdReal QRQR = QR * QR;
    const hkSimdReal Det = (QQRR - QRQR);

	if ( Det.isGreaterZero() )
	{
		hkSimdReal invDet; invDet.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(Det);

		hkVector4 relPos; relPos.setSub( t1, pos );
		const hkSimdReal sq = relPos.dot<3>(Q);
		const hkSimdReal sr = relPos.dot<3>(R);
    
		const hkSimdReal q = (sr * QR - RR * sq);
		const hkSimdReal r = (sq * QR - QQ * sr);
    
		lambdas.set(q,
					Det - q - r,
					r,
					hkSimdReal_0);
		lambdas.mul(invDet);
		return;
	}
	
	hkVector4 S;
	S.setSub( t0, t2 );
	const hkSimdReal SS = S.lengthSquared<3>();

	if ( QQ.isGreaterEqual(RR) )
	{
		if ( SS.isGreaterEqual(QQ) )
		{
			if ( SS.isGreaterZero() )
			{
				hkVector4 relPos; relPos.setSub( pos, t2 );
				const hkSimdReal p = relPos.dot<3>(S);
				hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,SS);
				lambdas.set(quot,
							hkSimdReal_0,
							hkSimdReal_1 - quot,
							hkSimdReal_0);
			}
			else
			{
				lambdas.setZero();
			}
		}
		else
		{
			hkVector4 relPos; relPos.setSub( pos, t1 );
			const hkSimdReal p  = relPos.dot<3>(Q);
			hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,QQ);
			lambdas.set(quot,
						hkSimdReal_1 - quot,
						hkSimdReal_0,
						hkSimdReal_0);
		}
	}
	else
	{
		if ( SS.isGreaterEqual(RR) )
		{
			hkVector4 relPos; relPos.setSub( pos, t2 );
			const hkSimdReal p = relPos.dot<3>(S);
			hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,SS);
			lambdas.set(quot,
						hkSimdReal_0,
						hkSimdReal_1 - quot,
						hkSimdReal_0);
		}
		else
		{
			hkVector4 relPos; relPos.setSub( pos, t1 );
			const hkSimdReal p  = relPos.dot<3>(R);
			hkSimdReal quot; quot.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(p,RR);
			lambdas.set(hkSimdReal_0,
						hkSimdReal_1 - quot,
						quot,
						hkSimdReal_0);
		}
	}
}


void hkLinearVertexCalculator::disableAllElements()
{	
	for (int i = 0; i < int(HK_COUNT_OF(m_epsilons)); i++)
	{
		// Disable all at first
		m_epsilons[i] = hkReal(-1);
	}
}

hkLinearVertexCalculator::~hkLinearVertexCalculator()
{
	HK_ASSERT(0x24234234, m_isStarted == false);
}

void hkLinearVertexCalculator::start()
{
	HK_ASSERT(0x24234234, m_isStarted == false);

	m_isStarted = true;
}

void hkLinearVertexCalculator::end()
{
	HK_ASSERT(0x24234234, m_isStarted );

	if (m_vertexBuffer)
	{
		m_vertexBuffer->unlock(m_lockedVertices);
		m_vertexBuffer = HK_NULL;
	}

	m_isStarted = false;
}


void hkLinearVertexCalculator::enableElement(hkVertexFormat::ComponentUsage usage, hkReal epsilon)
{
	HK_ASSERT(0x23423423, epsilon >= hkReal(0));

	m_epsilons[usage] = epsilon;
}

void hkLinearVertexCalculator::setVertexBuffer(hkMeshVertexBuffer* vertexBuffer)
{
	HK_ASSERT(0x2343a464, m_isStarted);

	// If has not changed, do nothing
	if (m_vertexBuffer == vertexBuffer)
	{
		return;
	}

	// Release if a vertex buffer is set
	if (m_vertexBuffer)
	{
		m_vertexBuffer->unlock(m_lockedVertices);
		m_vertexBuffer = HK_NULL;
	}

	// Work out what to lock based on 
	hkVertexFormat vertexFormat;
	vertexBuffer->getVertexFormat(vertexFormat);

	hkMeshVertexBuffer::LockInput input;
	input.m_lockFlags = hkMeshVertexBuffer::ACCESS_READ;

	// Find what elements need to be tested
	hkMeshVertexBuffer::PartialLockInput partialInput;
	partialInput.m_numLockFlags = 0;

	for (int i = 0; i < vertexFormat.m_numElements; i++)
	{
		hkReal epsilon = 0;
		if (checkElement(vertexFormat.m_elements[i], epsilon))
		{
			const int index = partialInput.m_numLockFlags++;

			partialInput.m_elementIndices[index] = i;
			partialInput.m_lockFlags[index] = hkUint8(hkMeshVertexBuffer::ACCESS_READ);

			m_lockedEpsilons[index] = epsilon;
		}
	}

	// If there is stuff to lock then try locking it
	if (partialInput.m_numLockFlags <= 0)
	{
		return;
	}

	// Lock the vertex buffer
	hkMeshVertexBuffer::LockResult res = vertexBuffer->partialLock(input, partialInput, m_lockedVertices);
	if (res == hkMeshVertexBuffer::RESULT_FAILURE)
	{
		return;
	}
	
	// Set the vertex buffer
	m_vertexBuffer = vertexBuffer;
	// Hold the positions
	m_rootTriangleValues.setSize(m_lockedVertices.m_numBuffers * 3);
}


hkBool hkLinearVertexCalculator::checkElement(const hkVertexFormat::Element& ele, hkReal& epsilonOut)
{
	hkReal epsilon = m_epsilons[ele.m_usage];

	if (epsilon >= 0)
	{
		epsilonOut = epsilon;
		return true;
	}
	return false;
}

void hkLinearVertexCalculator::setRootTriangle(const hkVector4 support[3], const int vertexIndices[3])
{
	HK_ASSERT(0x2343a464, m_isStarted);

	if (m_vertexBuffer == HK_NULL)
    {
		return;
	}

	m_support[0] = support[0];
	m_support[1] = support[1];
	m_support[2] = support[2];

	// Extract the values for the support
	for (int i = 0; i < m_lockedVertices.m_numBuffers; i++)
	{
		hkFloat32 fValues[4*3];
		hkMeshVertexBufferUtil::getIndexedElementVectorArray(m_lockedVertices.m_buffers[i], vertexIndices, &fValues[0], 3);

		m_rootTriangleValues[i * 3    ].load<4,HK_IO_NATIVE_ALIGNED>(&fValues[0*4]);
		m_rootTriangleValues[i * 3 + 1].load<4,HK_IO_NATIVE_ALIGNED>(&fValues[1*4]);
		m_rootTriangleValues[i * 3 + 2].load<4,HK_IO_NATIVE_ALIGNED>(&fValues[2*4]);
	}
}

hkBool hkLinearVertexCalculator::isLinear(hkVector4Parameter pos, int vertexIndex)
{
	HK_ASSERT(0x2343a464, m_isStarted);

	if (m_vertexBuffer == HK_NULL)
    {
		return true;
	}

    // Okay work out what value
    hkVector4 lambdas;
    calcBarycentricCoordinates(pos, m_support[0], m_support[1], m_support[2], lambdas);

	for (int i = 0; i < m_lockedVertices.m_numBuffers; i++)
	{
		hkFloat32 orgFval[4];
		hkMeshVertexBufferUtil::getIndexedElementVectorArray(m_lockedVertices.m_buffers[i], &vertexIndex, &orgFval[0], 1);
		hkVector4 originalValue; originalValue.load<4,HK_IO_NATIVE_ALIGNED>(&orgFval[0]);

		const hkVector4* supportValues = &m_rootTriangleValues[i * 3];

		// Work out the new vertex
		hkVector4 v;
		v.setMul( lambdas.getComponent<0>(), supportValues[0]);
		v.addMul( lambdas.getComponent<1>(), supportValues[1]);
		v.addMul( lambdas.getComponent<2>(), supportValues[2]);

		hkVector4 diff; diff.setSub(v, originalValue);
		const hkSimdReal linearErrorSquared = diff.lengthSquared<3>();

		hkSimdReal linearEpsilon; linearEpsilon.load<1>(&m_lockedEpsilons[i]);

		// Its too far
		if (linearErrorSquared > linearEpsilon * linearEpsilon)
		{
			return false;
		}
	}

	// It's linear
	return true;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
