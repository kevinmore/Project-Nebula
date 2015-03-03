/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//
HKGP_FORCE_INLINE bool		hkgpMesh::Edge::hasTetrahedralTopology() const
{
	const Edge	q=next().link();
	const Edge	w=prev().link();
	if(q.isValid()&&w.isValid())
	{
		if(q.apex()==w.apex())
		{
			return(true);
		}
	}
	return(false);
}
//
HKGP_FORCE_INLINE int			hkgpMesh::SingleEdge::hash() const
{
	int	xy[2]={*(const int*)&m_a,*(const int*)&m_b};
	if(xy[0]>xy[1]) hkAlgorithm::swap(xy[0],xy[1]);
	return(((xy[0]*45564901)^(xy[1]*95564881))%5564887);
}

//
HKGP_FORCE_INLINE bool			hkgpMesh::SingleEdge::operator==(const SingleEdge& x) const
{
	return(((x.m_a==m_a)&&(x.m_b==m_b))||((x.m_a==m_b)&&(x.m_b==m_a)));
}

//
HKGP_FORCE_INLINE bool		hkgpMesh::CollapseEdgePolicy::operator()(const Edge& e) const
{
	e.start()=m_newVertex;
	if(m_updatePlanes) m_mesh->updatePlane(e.triangle());
	return(true);
}

//
HKGP_FORCE_INLINE bool		hkgpMesh::FanEdgeCollector::operator()(const Edge& e)
{
	m_edges.pushBack(e);
	return(true);
}

//
HKGP_FORCE_INLINE bool		hkgpMesh::FanEdgeCollector::hasNakedEdges() const
{
	for(int i=0;i<m_edges.getSize();++i)
	{
		if(m_edges[i].isNaked()||m_edges[i].next().isNaked())
		{
			return(true);
		}
	}
	return(false);
}

//
HKGP_FORCE_INLINE bool		hkgpMesh::FanEdgeCollector::hasHeterogeneousPlaneIds() const
{
	if(m_edges.getSize())
	{
		const int planeId = m_edges[0].triangle()->m_planeId;
		for(int i=1;i<m_edges.getSize();++i)
		{
			if(m_edges[i].triangle()->m_planeId!=planeId) return true;
		}
		return false;
	}
	return false;
}

//
HKGP_FORCE_INLINE int		hkgpMesh::FanEdgeCollector::countPlanes() const
{
	hkPointerMap<int,int>	planes; planes.reserve(16);
	for(int i=0;i<m_edges.getSize();++i)
	{
		if(!planes.getWithDefault(m_edges[i].triangle()->m_planeId,0))
		{
			planes.insert(m_edges[i].triangle()->m_planeId,1);
		}
	}
	return planes.getSize();
}

//
HKGP_FORCE_INLINE void		hkgpMesh::FanEdgeCollector::setVerticesTag(int id) const
{
	for(int i=0;i<m_edges.getSize();++i)
	{
		m_edges[i].end()->m_tag=id;
	}
}

//
HKGP_FORCE_INLINE void		hkgpMesh::FanEdgeCollector::incVerticesTag(int id) const
{
	for(int i=0;i<m_edges.getSize();++i)
	{
		m_edges[i].end()->m_tag+=id;
	}
}

//
HKGP_FORCE_INLINE int			hkgpMesh::FanEdgeCollector::countVerticesTag(int id) const
{
	int num=0;
	for(int i=0;i<m_edges.getSize();++i)
	{
		if(m_edges[i].end()->m_tag==id) ++num;
	}
	return(num);
}

//
HKGP_FORCE_INLINE				hkgpMesh::CollapseMetric::CollapseMetric(hkgpMesh* mesh, hkVector4Parameter position) :
	m_mesh(mesh),
	m_position(position),
	m_minDot(hkSimdReal_1),
	m_srcArea(hkSimdReal_0),
	m_dstArea(hkSimdReal_0)
{}

//
HKGP_FORCE_INLINE bool		hkgpMesh::CollapseMetric::operator()(Edge edge)
{
	hkVector4	srcPlane;
	hkVector4	dstPlane;
	hkSimdReal	dstArea;

	m_srcArea.add(m_mesh->getTwiceArea(edge.triangle()));
	m_mesh->setPlane(edge.triangle(), srcPlane, true);

	hkVector4	backup=edge.start()->m_position;
	edge.start()->m_position=m_position;
	dstArea = m_mesh->getTwiceArea(edge.triangle());
	m_mesh->setPlane(edge.triangle(), dstPlane, true);
	edge.start()->m_position=backup;

	if(dstArea.isGreater(hkSimdReal_Eps))
	{
		m_minDot.setMin(m_minDot,srcPlane.dot<3>(dstPlane));
		m_dstArea.add(dstArea);
	}
	return true;
}

//
HKGP_FORCE_INLINE hkSimdReal		hkgpMesh::getTwiceArea(const Triangle* triangle)
{
	hkVector4	ab,ac;
	ab.setSub(triangle->vertex(1)->m_position,triangle->vertex(0)->m_position);
	ac.setSub(triangle->vertex(2)->m_position,triangle->vertex(0)->m_position);
	hkVector4	crs;
	crs.setCross(ab,ac);
	return crs.length<3>();
}

//
template <typename T>
HKGP_FORCE_INLINE int		hkgpMesh::floodFill(T& boundaries)
{
	int	partId=0;
	/* Cleanup	*/ 
	for(Triangle* t=m_triangles.getFirst();t;t=t->next())
	{
		t->m_partId=-1;
	}
	/* Fill		*/ 
	hkArray<Edge>	stack;
	for(Triangle* t=m_triangles.getFirst();t;t=t->next())
	{
		if(t->m_partId==-1)
		{
			stack.clear();
			stack.pushBack(Edge(t,0));
			stack.pushBack(Edge(t,1));
			stack.pushBack(Edge(t,2));
			t->m_partId=partId++;
			do
			{
				Edge	e=stack.back();
				Edge	l=e.link();
				stack.popBack();
				if(l.isValid()&&(l.triangle()->m_partId==-1)&&boundaries.canCrossEdge(this,e))
				{
					l.triangle()->m_partId=partId-1;
					stack.pushBack(l.next());
					stack.pushBack(l.prev());
				}
			} while(stack.getSize()!=0);
		}
	}
	return(partId);
}

//
HKGP_FORCE_INLINE				hkgpMesh::ExtrudeShape::ExtrudeShape(Triangle* t,hkReal width) : IConvexOverlap::IConvexShape()
{
	hkInplaceArray<hkVector4,6>	vertices;
	vertices.setSize(6);
	vertices[0]=t->vertex(0)->m_position;
	vertices[1]=t->vertex(1)->m_position;
	vertices[2]=t->vertex(2)->m_position;
	hkSimdReal swidth; swidth.setFromFloat(width);
	vertices[3].setAddMul(vertices[0],t->vertex(0)->m_normal,swidth);
	vertices[4].setAddMul(vertices[1],t->vertex(1)->m_normal,swidth);
	vertices[5].setAddMul(vertices[2],t->vertex(2)->m_normal,swidth);
	hkGeometryProcessing::buildTransposedArray(vertices,m_vertices);
}

//
HKGP_FORCE_INLINE void			hkgpMesh::ExtrudeShape::getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const
{
	hkGeometryProcessing::getSupportingVertex(m_vertices,direction,out);
}

//
HKGP_FORCE_INLINE hkAabb		hkgpMesh::ExtrudeShape::getBoundingBox() const
{
	HK_ERROR(0xB6C47B76,"Unsupported operation");
	hkAabb aabb;
	aabb.setEmpty();
	return(aabb);
}

//
HKGP_FORCE_INLINE void			hkgpMesh::TriangleShape::getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const
{
	const hkVector4	vtx[]={m_triangle->vertex(0)->m_position,m_triangle->vertex(1)->m_position,m_triangle->vertex(2)->m_position};
	hkVector4 dots;
	hkVector4Util::dot3_1vs3(direction, vtx[0], vtx[1], vtx[2], dots);
	const int index = dots.getIndexOfMaxComponent<3>();
	out = vtx[index];
	out.setInt24W(index);
}

//
HKGP_FORCE_INLINE hkAabb		hkgpMesh::TriangleShape::getBoundingBox() const
{
	const hkVector4			x[]={m_triangle->vertex(0)->m_position,m_triangle->vertex(1)->m_position,m_triangle->vertex(2)->m_position};
	hkAabb					aabb;
	hkAabbUtil::calcAabb(x,3,aabb);
	return aabb;
}

//
HKGP_FORCE_INLINE hkAabb		hkgpMesh::ExternShape::getBoundingBox() const
{
	hkAabb	aabb;
	hkAabbUtil::calcAabb(m_vertices,m_numVertices,aabb);
	return aabb;
}

//
HKGP_FORCE_INLINE void			hkgpMesh::ExternShape::getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const
{
	int			bestV=0;
	hkSimdReal	bestD=direction.dot<3>(m_vertices[0]);
	for(int i=1;i<m_numVertices;++i)
	{
		const hkSimdReal	d=direction.dot<3>(m_vertices[i]);
		if(d>bestD)
		{
			bestD	=	d;
			bestV	=	i;
		}
	}
	out	=	m_vertices[bestV];
	out.setInt24W(bestV);
}

//
HKGP_FORCE_INLINE void		hkgpMesh::ConvexHullShape::getSupportingVertex(hkVector4Parameter direction,hkVector4& output) const
{
	m_hull->getSupportingVertex(direction,output);
}

//
HKGP_FORCE_INLINE hkAabb		hkgpMesh::ConvexHullShape::getBoundingBox() const
{
	return(m_hull->getBoundingBox(hkgpConvexHull::SOURCE_VERTICES,1,0));
}


HK_DISABLE_OPTIMIZATION_VS2008_X64
HKGP_FORCE_INLINE void	hkgpMesh::LineShape::getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const
{

	// X64BUGFIX  
	out.setSelect(direction.dot<3>(m_a).greater(direction.dot<3>(m_b)),m_a,m_b);
	if(m_radius.isGreaterZero())
	{
		hkVector4	n = direction;
		const hkSimdReal	w = out.getW();
		n.normalizeIfNotZero<3>();
		out.addMul(m_radius,n);
		out.setW(w);
	}
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
