/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Mesh/hkxIndexBuffer.h>

/// Returns the number of triangles
int hkxIndexBuffer::getNumTriangles () const
{
    int numIndices = m_indices16.getSize() ? m_indices16.getSize() : m_indices32.getSize();

    switch ( m_indexType )
    {
        case hkxIndexBuffer::INDEX_TYPE_TRI_LIST:
        {
            return numIndices/3;
        }

        case hkxIndexBuffer::INDEX_TYPE_TRI_STRIP:
        {
            if (numIndices > 2)
            {
              return numIndices - 2;
            }
            break;
        }

        default:
            break;
    }

    return 0;
}

bool hkxIndexBuffer::getTriangleIndices (hkUint32 triIndex, hkUint32& indexAOut, hkUint32& indexBOut, hkUint32& indexCOut)
{
    if ( m_indices16.getSize() )
    {
      // 16 bit indices
      switch ( m_indexType )
      {
        case hkxIndexBuffer::INDEX_TYPE_TRI_LIST:
        {
          HK_ON_DEBUG( if ((triIndex*3 + 2) < (hkUint32)m_indices16.getSize()) )
          {
            indexAOut = (hkUint32)m_indices16[triIndex*3];
            indexBOut = (hkUint32)m_indices16[triIndex*3+1];
            indexCOut = (hkUint32)m_indices16[triIndex*3+2];
            return true;
          }
          break;
        }

        case hkxIndexBuffer::INDEX_TYPE_TRI_STRIP:
        {
          HK_ON_DEBUG( if ((triIndex + 2) < (hkUint32)m_indices16.getSize()) )
          {
            indexAOut = (hkUint32)m_indices16[triIndex];
            indexBOut = (hkUint32)m_indices16[triIndex+1];
            indexCOut = (hkUint32)m_indices16[triIndex+2];
            return true;
          }
          break;
        }

      default: break;
      }
    }
    else if ( m_indices32.getSize() )
    {
      // 32 bit indices
      switch ( m_indexType )
      {
        case hkxIndexBuffer::INDEX_TYPE_TRI_LIST:
        {
          HK_ON_DEBUG( if ((triIndex*3 + 2) < (hkUint32)m_indices32.getSize()) )
          {
            indexAOut = m_indices32[triIndex*3];
            indexBOut = m_indices32[triIndex*3+1];
            indexCOut = m_indices32[triIndex*3+2];
            return true;
          }
          break;
        }

        case hkxIndexBuffer::INDEX_TYPE_TRI_STRIP:
        {
          HK_ON_DEBUG( if ((triIndex + 2) < (hkUint32)m_indices32.getSize()) )
          {
            indexAOut = m_indices32[triIndex];
            indexBOut = m_indices32[triIndex+1];
            indexCOut = m_indices32[triIndex+2];
            return true;
          }
          break;
        }

        default: break;
      }
    }

	indexAOut = HKX_INDEX_BUFFER_INVALID_INDEX;
	indexBOut = HKX_INDEX_BUFFER_INVALID_INDEX; 
	indexCOut = HKX_INDEX_BUFFER_INVALID_INDEX;
	return false;
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
