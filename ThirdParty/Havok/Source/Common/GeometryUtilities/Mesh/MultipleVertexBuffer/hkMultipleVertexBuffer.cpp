/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/MultipleVertexBuffer/hkMultipleVertexBuffer.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>

hkMultipleVertexBuffer::hkMultipleVertexBuffer(const hkVertexFormat& format, int numVertices)
{
    m_vertexFormat = format;
    m_numVertices = numVertices;
    m_isLocked = false;

    m_updateCount = 1;
	m_constructionComplete = false;

	m_lockedElements.setSize(format.m_numElements);
}

hkMultipleVertexBuffer::hkMultipleVertexBuffer( hkFinishLoadedObjectFlag flag )
:hkMeshVertexBuffer(flag)
, m_vertexFormat(flag)
, m_lockedElements(flag)
, m_lockedBuffer(flag)
, m_elementInfos(flag)
, m_vertexBufferInfos(flag)
{

}

hkMultipleVertexBuffer::hkMultipleVertexBuffer(const hkMultipleVertexBuffer& rhs)
	: hkMeshVertexBuffer()
{
    m_isLocked = false;
    m_numVertices = rhs.m_numVertices;
    m_vertexFormat = rhs.m_vertexFormat;

    m_elementInfos = rhs.m_elementInfos;
    m_vertexBufferInfos = rhs.m_vertexBufferInfos;

    const int numBuffers = m_vertexBufferInfos.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        VertexBufferInfo& info = m_vertexBufferInfos[i];

        info.m_vertexBuffer = info.m_vertexBuffer->clone();
        info.m_vertexBuffer->removeReference();
    }

	m_lockedElements.setSize(m_vertexFormat.m_numElements);

	m_constructionComplete = false;
}

hkMeshVertexBuffer* hkMultipleVertexBuffer::clone()
{
	HK_ASSERT(0x24532423, m_constructionComplete);
	if (!m_isSharable)
	{
		hkMultipleVertexBuffer* newBuffer = new hkMultipleVertexBuffer(*this);
		newBuffer->completeConstruction();
		return newBuffer;
	}
	// If they are all shared, we can just use this
	addReference();
	return this;
}

hkMeshVertexBuffer::LockResult hkMultipleVertexBuffer::lock( const LockInput& inputIn, LockedVertices& lockedVerticesOut )
{
	HK_ASSERT(0x24532423, m_constructionComplete);

    if (m_isLocked)
    {
        HK_ASSERT(0x193be906, !"Already locked");
        return RESULT_FAILURE;
    }

    // Mark if its a write lock
    m_writeLock = (inputIn.m_lockFlags & hkMeshVertexBuffer::ACCESS_WRITE) != 0;

    // 1st I need to find every buffer that needs to be locked
    const int numElements = m_vertexFormat.m_numElements;

    // Set one bigger as we'll use this to speed up ordering
    m_lockedElements.setSizeUnchecked(numElements);

    for (int i = 0; i < numElements; i++)
    {
        const ElementInfo& element = m_elementInfos[i];

        LockedElement& lockedElement = m_lockedElements[i];

        lockedElement.m_vertexBufferIndex = element.m_vertexBufferIndex;
        lockedElement.m_elementIndex = element.m_elementIndex;

        lockedElement.m_lockFlags = hkUint8(inputIn.m_lockFlags);

        lockedElement.m_outputBufferIndex = hkUint8(i);
        lockedElement.m_vertexFormatIndex = hkUint8(i);
    }	

    return _lockElements(inputIn, lockedVerticesOut);
}

void hkMultipleVertexBuffer::_unlockVertexBuffers()
{
    const int numBuffers = m_vertexBufferInfos.getSize();
    for (int i = 0; i < numBuffers; i++)
    {
        VertexBufferInfo& bufferInfo = m_vertexBufferInfos[i];
        if (bufferInfo.m_isLocked)
        {            
			bufferInfo.m_vertexBuffer->unlock(*bufferInfo.m_lockedVertices);
            bufferInfo.m_isLocked = false;
        }
		if (bufferInfo.m_lockedVertices)
		{
			delete bufferInfo.m_lockedVertices;
			bufferInfo.m_lockedVertices = HK_NULL;
		}
    }
}

hkMeshVertexBuffer::LockResult hkMultipleVertexBuffer::partialLock( const LockInput& input, const PartialLockInput& partialInput, LockedVertices& lockedOut)
{
	HK_ASSERT(0x24532423, m_constructionComplete);
    if (m_isLocked)
    {
        HK_ASSERT(0x193be907, !"Already locked");
        return RESULT_FAILURE;
    }

    // 1st I need to find every buffer that needs to be locked
    const int numElements = partialInput.m_numLockFlags;

    // Set one bigger as we'll use this to speed up ordering
    m_lockedElements.setSizeUnchecked(numElements);

    int lockFlags = 0;
    for (int i = 0; i < numElements; i++)
    {
        const ElementInfo& element = m_elementInfos[partialInput.m_elementIndices[i]];

        LockedElement& lockedElement = m_lockedElements[i];

        lockedElement.m_vertexBufferIndex = element.m_vertexBufferIndex;
        lockedElement.m_elementIndex = element.m_elementIndex;

        lockFlags |= partialInput.m_lockFlags[i];
        lockedElement.m_lockFlags = partialInput.m_lockFlags[i];

        lockedElement.m_outputBufferIndex = hkUint8(i);
        lockedElement.m_vertexFormatIndex = hkUint8(partialInput.m_elementIndices[i]);
    }

    // Mark if its a write lock
    m_writeLock = (lockFlags & hkMeshVertexBuffer::ACCESS_WRITE) != 0;

    return _lockElements(input, lockedOut);
}

hkMeshVertexBuffer::LockResult hkMultipleVertexBuffer::_lockElements( const hkMeshVertexBuffer::LockInput& input, hkMeshVertexBuffer::LockedVertices& lockedOut)
{
	// Allocate temporary LockedVertices to store buffer information
	for( int i = 0; i < m_vertexBufferInfos.getSize(); ++i )
	{
		m_vertexBufferInfos[i].m_lockedVertices = new hkMeshVertexBuffer::LockedVertices();
	}

    const int numElements = m_lockedElements.getSize();
    const int numVertices = input.m_numVertices < 0 ? m_numVertices - input.m_startVertex : input.m_numVertices;

    hkSort(m_lockedElements.begin(), m_lockedElements.getSize(), _less);

    {
        LockedElement* start = m_lockedElements.begin();
        LockedElement* end = m_lockedElements.end();

        while (start < end)
        {
            int vertexBufferIndex = start->m_vertexBufferIndex;

            LockedElement* cur = start + 1;
            while (cur < end && cur->m_vertexBufferIndex == vertexBufferIndex)
            {
				// We must have unique element indices
				HK_ASSERT(0x3424234, cur[-1].m_elementIndex != cur[0].m_elementIndex);
				// Next
                cur++;
            }

            VertexBufferInfo& bufferInfo = m_vertexBufferInfos[vertexBufferIndex];
            // Okay we have everything we need to set up a lock
            const int numLockElements = int(cur - start);

            {
				hkMeshVertexBuffer::PartialLockInput partialInput;

                partialInput.m_numLockFlags = numLockElements;
                for (int i = 0; i < numLockElements; i++)
                {
                    partialInput.m_elementIndices[i] = start[i].m_elementIndex;
                    partialInput.m_lockFlags[i] = start[i].m_lockFlags;
                    start[i].m_lockedBufferIndex = hkUint8(i);
                }
                
				if( bufferInfo.m_lockedVertices == HK_NULL )
				{
					bufferInfo.m_lockedVertices = new hkMeshVertexBuffer::LockedVertices();
				}

				hkMeshVertexBuffer::LockResult res = bufferInfo.m_vertexBuffer->partialLock(input, partialInput, *bufferInfo.m_lockedVertices);

                // If didn't lock correctly - just unlock all
                if (res != hkMeshVertexBuffer::RESULT_SUCCESS)
                {
                    _unlockVertexBuffers();
                    return res;
                }
            }

			// It is now locked
			bufferInfo.m_isLocked = true;

            // Next
            start = cur;
        }
    }

    // Set up
    lockedOut.m_numBuffers = numElements;
    lockedOut.m_numVertices = numVertices;
    lockedOut.m_isInterleaved = false;

    // Okay we may need to emulate some members
    hkVertexFormat emulateFormat;
    for (int i = 0; i < numElements; i++)
    {
        LockedElement& lockedElement = m_lockedElements[i];
        VertexBufferInfo& bufferInfo = m_vertexBufferInfos[lockedElement.m_vertexBufferIndex];

        hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer = bufferInfo.m_lockedVertices->m_buffers[lockedElement.m_lockedBufferIndex];
		hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer = lockedOut.m_buffers[lockedElement.m_outputBufferIndex];

        const hkVertexFormat::Element& srcElement = srcBuffer.m_element;
        const hkVertexFormat::Element& dstElement = m_vertexFormat.m_elements[lockedElement.m_vertexFormatIndex];

        if (dstElement.m_dataType != srcElement.m_dataType || dstElement.m_numValues != srcElement.m_numValues)
        {
            if ((srcElement.m_dataType == hkVertexFormat::TYPE_VECTOR4 && srcElement.m_numValues == 1 &&
				 dstElement.m_dataType == hkVertexFormat::TYPE_FLOAT32 && dstElement.m_numValues <= 4))
            {
                // Don't emulate - if the vector can look like 4 consecutive floats
                dstBuffer = srcBuffer;

                // Make the type look like the dst type
                hkVertexFormat::Element& ele = dstBuffer.m_element;
                ele.m_dataType = dstElement.m_dataType;
                ele.m_numValues = dstElement.m_numValues;

                lockedElement.m_emulatedIndex = -1;
            }
            else
            {
                // Looks like we need to emulate this member
                emulateFormat.addElement(dstElement);
                lockedElement.m_emulatedIndex = 0;

				// Set the type
				dstBuffer.m_element = dstElement;
            }
        }
        else
        {
            // Set the buffer
            dstBuffer = srcBuffer;
            lockedElement.m_emulatedIndex = -1;
        }
    }
    if (emulateFormat.m_numElements <= 0)
    {
        m_isLocked = true;
        return RESULT_SUCCESS;
    }

    //
    emulateFormat.makeCanonicalOrder();
    m_lockedBuffer = new hkMemoryMeshVertexBuffer(emulateFormat, numVertices);
	m_lockedBuffer->removeReference();

    for (int i = 0; i < numElements; i++)
    {
        LockedElement& lockedElement = m_lockedElements[i];

        if (lockedElement.m_emulatedIndex >= 0)
        {
            const hkVertexFormat::Element& dstElement = m_vertexFormat.m_elements[lockedElement.m_vertexFormatIndex];
            lockedElement.m_emulatedIndex = hkInt8(emulateFormat.findElementIndex(dstElement.m_usage, dstElement.m_subUsage));
            HK_ASSERT(0x3422342, lockedElement.m_emulatedIndex >= 0);

			// We need to copy it over
			hkMeshVertexBuffer::LockedVertices::Buffer dstBuffer;
			// Get the buffer
			m_lockedBuffer->getLockedVerticesBuffer(lockedElement.m_emulatedIndex, dstBuffer);

            // Okay. Do we need to copy data over
			if ((lockedElement.m_lockFlags & hkMeshVertexBuffer::ACCESS_WRITE_DISCARD) == 0)
            {
                VertexBufferInfo& bufferInfo = m_vertexBufferInfos[lockedElement.m_vertexBufferIndex];
                hkMeshVertexBuffer::LockedVertices::Buffer& srcBuffer = bufferInfo.m_lockedVertices->m_buffers[lockedElement.m_lockedBufferIndex];

                // Need to do a copy
                hkMeshVertexBufferUtil::convert(srcBuffer, dstBuffer, numVertices);
            }

			// We need to set the buffer
			hkMeshVertexBuffer::LockedVertices::Buffer& lockedBuffer = lockedOut.m_buffers[lockedElement.m_outputBufferIndex];
			lockedBuffer.m_start = dstBuffer.m_start;
			lockedBuffer.m_stride = dstBuffer.m_stride;
        }
    }

    m_isLocked = true;
    return RESULT_SUCCESS;
}

void hkMultipleVertexBuffer::unlock(const LockedVertices& lockedVertices)
{
	HK_ASSERT(0x24532423, m_constructionComplete);
    if (!m_isLocked)
    {
        HK_ASSERT(0x32424, !"The buffer isn't locked");
        return;
    }

	if (m_lockedBuffer)
    {
        const int numVertices = m_lockedBuffer->getNumVertices();

        // If anything is emulated, I need to copy the data back from the the locked buffer into the
        // actual buffer it originated from
        const int numBuffers = m_lockedElements.getSize();

        for (int i = 0; i < numBuffers; i++)
        {
            LockedElement& lockedElement = m_lockedElements[i];
			if ((lockedElement.m_lockFlags & hkMeshVertexBuffer::ACCESS_WRITE) != 0 && lockedElement.m_emulatedIndex >= 0)
            {
                // We need to do a copy back

                hkMeshVertexBuffer::LockedVertices::Buffer srcBuffer;
                m_lockedBuffer->getLockedVerticesBuffer(lockedElement.m_emulatedIndex, srcBuffer);

                VertexBufferInfo& bufferInfo = m_vertexBufferInfos[lockedElement.m_vertexBufferIndex];
                hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer = bufferInfo.m_lockedVertices->m_buffers[lockedElement.m_lockedBufferIndex];

                // Need to do a copy
                hkMeshVertexBufferUtil::convert(srcBuffer, dstBuffer, numVertices);
            }
        }

        // We don't need the locked buffer anymore
        m_lockedBuffer = HK_NULL;		
    }

	_unlockVertexBuffers();

    if (m_writeLock)
    {
        m_updateCount ++;
        if (m_updateCount == 0) m_updateCount++;
    }

    m_isLocked = false;
}

void hkMultipleVertexBuffer::getElementVectorArray(const LockedVertices& lockedVertices, int elementIndex, hkFloat32* data)
{
	hkMeshVertexBufferUtil::getElementVectorArray(lockedVertices, elementIndex, data);
}

void hkMultipleVertexBuffer::setElementVectorArray(const LockedVertices& lockedVertices, int elementIndex, const hkFloat32* data)
{
	hkMeshVertexBufferUtil::setElementVectorArray(lockedVertices, elementIndex, data);
}

void hkMultipleVertexBuffer::getElementIntArray(const LockedVertices& lockedVertices, int elementIndex, int* data)
{
    hkMeshVertexBufferUtil::getElementIntArray(lockedVertices, elementIndex, data);
}

void hkMultipleVertexBuffer::setElementIntArray(const LockedVertices& lockedVertices, int elementIndex, const int* data)
{
    hkMeshVertexBufferUtil::setElementIntArray(lockedVertices, elementIndex, data);
}

void hkMultipleVertexBuffer::addElement(int vertexBufferIndex, int elementIndex)
{
	HK_ASSERT(0x24532423, !m_constructionComplete);
    ElementInfo& elementInfo = m_elementInfos.expandOne();

    elementInfo.m_vertexBufferIndex = hkUint8(vertexBufferIndex);
    elementInfo.m_elementIndex = hkUint8(elementIndex);
}

void hkMultipleVertexBuffer::addVertexBuffer(hkMeshVertexBuffer* vertexBuffer)
{
	HK_ASSERT(0x24532423, !m_constructionComplete);
    VertexBufferInfo& info = m_vertexBufferInfos.expandOne();
    HK_ASSERT(0xbd838ddc, vertexBuffer->getNumVertices() >= m_numVertices);
    info.m_vertexBuffer = vertexBuffer;
}

void hkMultipleVertexBuffer::completeConstruction()
{
	HK_ASSERT(0x24532423, !m_constructionComplete);

	m_isSharable = true;
	for (int i = 0; i < m_vertexBufferInfos.getSize(); i++)
	{
		if (!m_vertexBufferInfos[i].m_vertexBuffer->isSharable())
		{
			m_isSharable = false;
			break;
		}
	}
	m_constructionComplete = true;
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
