/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/IndexedTransformSet/hkIndexedTransformSet.h>
#include <Common/Base/Reflection/hkClass.h>

hkIndexedTransformSetCinfo::hkIndexedTransformSetCinfo()
: m_inverseMatrices(HK_NULL)
, m_matrices(HK_NULL)
, m_matricesOrder(HK_NULL)
, m_matricesNames(HK_NULL)
, m_numMatrices(0)
, m_allTransformsAreAffine(true)
, m_indexMappings(HK_NULL)
, m_numIndexMappings(0)
{ 
}

hkIndexedTransformSet::hkIndexedTransformSet( hkFinishLoadedObjectFlag flag )
: hkReferencedObject(flag)
, m_matrices(flag)
, m_inverseMatrices(flag)
, m_matricesOrder(flag)
, m_matricesNames(flag)
, m_indexMappings(flag)
{
}

hkIndexedTransformSet::hkIndexedTransformSet(const hkIndexedTransformSetCinfo& info)
{
	const int numMatrices = info.m_numMatrices;

    m_matrices.reserveExactly(numMatrices);
	m_matrices.setSizeUnchecked(numMatrices);

	HK_COMPILE_TIME_ASSERT(sizeof(hkMatrix4) % 16 == 0);

    // Set as identity or
    if (info.m_matrices == HK_NULL)
	{
		for (int i = 0; i < numMatrices; i++)
		{
			m_matrices[i].setIdentity();
		}
	}
    else
    {
        hkString::memCpy16(m_matrices.begin(), info.m_matrices, sizeof(hkMatrix4) * numMatrices / 16);
    }
    if (info.m_inverseMatrices != HK_NULL)
    {
        m_inverseMatrices.reserveExactly(numMatrices);
        m_inverseMatrices.setSizeUnchecked(numMatrices);

		hkString::memCpy16(m_inverseMatrices.begin(), info.m_inverseMatrices, sizeof(hkMatrix4) * numMatrices / 16);
	}

	// Record this... for skinning
	m_allMatricesAreAffine = info.m_allTransformsAreAffine;

	// Store optional transform order
	if (info.m_matricesOrder != HK_NULL)
	{
		m_matricesOrder.reserveExactly(numMatrices);
		m_matricesOrder.setSizeUnchecked(numMatrices);
		hkString::memCpy(m_matricesOrder.begin(), info.m_matricesOrder, sizeof(hkInt16) * numMatrices);
	}
		
	// Store optional transform names
	if (info.m_matricesNames != HK_NULL)
	{
		m_matricesNames.reserveExactly(numMatrices);
		m_matricesNames.setSizeUnchecked(numMatrices);
		for( hkInt32 i = 0; i < numMatrices; ++i )
		{
			m_matricesNames[i] = info.m_matricesNames[i];
		}		
	}

	// Store optional index mappings
	if (info.m_indexMappings != HK_NULL)
	{
		m_indexMappings.reserveExactly(info.m_numIndexMappings);
		m_indexMappings.setSizeUnchecked(info.m_numIndexMappings);

		for( hkInt32 i = 0; i < info.m_numIndexMappings; ++i )
		{
			m_indexMappings[i] = info.m_indexMappings[i];			
		}
	}
}

void hkIndexedTransformSet::setMatrices(int startIndex, const hkMatrix4* matrices, int numMatrices)
{
    HK_ASSERT(0x324242, startIndex >= 0 && startIndex < m_matrices.getSize() && numMatrices >= 0 && startIndex + numMatrices <= m_matrices.getSize());
    if (numMatrices <=0)
    {
        return;
    }

    hkString::memCpy16(m_matrices.begin() + startIndex, matrices, sizeof(hkMatrix4) / 16 * numMatrices);
}

void hkIndexedTransformSet::calculateMatrices(hkArray<hkMatrix4>& matricesOut) const
{
    const int numMatrices = getNumMatrices();
    matricesOut.setSize(numMatrices);

    const hkMatrix4* boneToModel = m_matrices.begin();
    if (hasInverseMatrices())
    {
        const hkMatrix4* localToBone = m_inverseMatrices.begin();
        for (int i = 0; i < numMatrices; i++)
        {
            matricesOut[i].setMul(boneToModel[i], localToBone[i]);
        }
    }
    else
    {
        // We can just copy the matrices out
        hkString::memCpy16(matricesOut.begin(), boneToModel, sizeof(hkMatrix4) / 16 * numMatrices);
    }
}

void hkIndexedTransformSet::calculateMatrices(const hkMatrix4& parentToWorld, hkArray<hkMatrix4>& matricesOut) const
{
    const int numMatrices = getNumMatrices();
    matricesOut.setSize(numMatrices);
    hkMatrix4* dstMatrices = matricesOut.begin();

    const hkMatrix4* boneToModel = m_matrices.begin();

    hkBool32 parentToWorldIsIdentity = parentToWorld.isApproximatelyIdentity(hkSimdReal::fromFloat(1e-5f));

    if (parentToWorldIsIdentity)
    {
        if (hasInverseMatrices())
        {
            const hkMatrix4* localToBone = m_inverseMatrices.begin();
            for (int i = 0; i < numMatrices; i++)
            {
                dstMatrices[i].setMul(boneToModel[i], localToBone[i]);
            }
        }
        else
        {
            // We can just copy the matrices out
            hkString::memCpy16(dstMatrices, boneToModel, sizeof(hkMatrix4) / 16 * numMatrices);
        }
    }
    else
    {
        if (hasInverseMatrices())
        {
            const hkMatrix4* localToBone = m_inverseMatrices.begin();
            hkMatrix4 tmp;

            for (int i = 0; i < numMatrices; i++)
            {
                tmp.setMul(parentToWorld, boneToModel[i]);
                dstMatrices[i].setMul(tmp, localToBone[i]);
            }
        }
        else
        {
            //const hkMatrix4* localToBone = m_inverseMatrices.begin();
            for (int i = 0; i < numMatrices; i++)
            {
                dstMatrices[i].setMul(parentToWorld, boneToModel[i]);
            }
        }
    }
}


void hkIndexedTransformSet::calculateMatrix(int index, hkMatrix4& matrixOut) const
{
    const hkMatrix4* localToBone = m_inverseMatrices.begin();
    const hkMatrix4* boneToModel = m_matrices.begin();
    if (hasInverseMatrices())
    {
        matrixOut.setMul(boneToModel[index], localToBone[index]);
    }
    else
    {
        matrixOut = boneToModel[index];
    }
}

void hkIndexedTransformSet::calculateMatrix(const hkMatrix4& parentToWorld, int index, hkMatrix4& matrixOut) const
{
    const hkMatrix4* localToBone = m_inverseMatrices.begin();
    const hkMatrix4* boneToModel = m_matrices.begin();
    if (hasInverseMatrices())
    {
        hkMatrix4 tmp;
        tmp.setMul(parentToWorld, boneToModel[index]);
        matrixOut.setMul(tmp, localToBone[index]);
    }
    else
    {
        matrixOut.setMul(parentToWorld, boneToModel[index]);
    }
}


void hkIndexedTransformSet::getMatrices(int startIndex, hkMatrix4* matrices, int numMatrices) const
{
    HK_ASSERT(0x324242, startIndex >= 0 && startIndex < m_matrices.getSize() && numMatrices >= 0 && startIndex + numMatrices <= m_matrices.getSize());
    if (numMatrices <=0)
    {
        return;
    }

    hkString::memCpy16(matrices, m_matrices.begin() + startIndex, sizeof(hkMatrix4) / 16 * numMatrices);

}

void hkIndexedTransformSet::getInverseMatrices(int startIndex, hkMatrix4* matrices, int numMatrices) const
{
    if (hasInverseMatrices())
    {
        HK_ASSERT(0x324242, startIndex >= 0 && startIndex < m_inverseMatrices.getSize() && numMatrices >= 0 && startIndex + numMatrices <= m_inverseMatrices.getSize());
        if (numMatrices <=0)
        {
            return;
        }

        hkString::memCpy16(matrices, m_inverseMatrices.begin() + startIndex, sizeof(hkMatrix4) / 16 * numMatrices);
    }
    else
    {
        for (int i = 0; i < numMatrices; i++)
        {
            matrices[i].setIdentity();
        }
    }
}

void hkIndexedTransformSet::setInverseMatrices( int startIndex, const hkMatrix4* matrices, int numMatrices )
{
	if ( !hasInverseMatrices() )
	{
		return;
	}

	HK_ASSERT(0x324242, startIndex >= 0 && startIndex < m_inverseMatrices.getSize() && numMatrices >= 0 && startIndex + numMatrices <= m_inverseMatrices.getSize());
	if (numMatrices <=0)
	{
		return;
	}

	hkString::memCpy16(m_inverseMatrices.begin() + startIndex, matrices, sizeof(hkMatrix4) / 16 * numMatrices);
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
