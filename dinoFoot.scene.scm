(let ((home (getenv "HOME")))
 (add-model "/tmp/boxes-d3.obj" :type :obj :is-base #t)
 (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/bc_body_trexAA.ptx"
  :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/bc_body_trexAA.ptx"
  :proxy "/tmp/trex.obj")
(add-model "/tmp/ram/trex/COLOR/lr_1claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_1claw_trexAA.ptx")
 (add-model "/tmp/ram/trex/COLOR/lr_2claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_2claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/lr_3claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_3claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/lr_4claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_4claw_trexAA.ptx")
 (subd-tess 3 7) ;use normal subdiv for 3 levels, quantization for 5 levels.
 (displacement-scale 4)
 (path-length 2)
 (path-samples 32)
 (light-samples 32)
 (use-only-those-faces "0,1552:0,1553:0,1554:0,1555:0,1556:0,1557:0,1563:0,1564:0,1565:0,1566:0,1567:0,1568:0,1569:0,1570:0,1576:0,1577:0,1578:0,1579:0,1580:0,1581:0,1582:0,1583:0,1584:0,1591:0,1592:0,1593:0,1594:0,1595:0,1596:0,1597:0,1598:0,1599:0,1600:0,1601:0,1602:0,1607:0,1608:0,1609:0,1610:0,1611:0,1612:0,1613:0,1614:0,1615:0,1616:0,1617:0,1618:0,1627:0,1628:0,1629:0,1630:0,1631:0,1632:0,1633:0,1634:0,1635:0,1636:0,1637:0,1638:0,1639:0,1651:0,1652:0,1653:0,1654:0,1660:0,1663:0,1664:0,1665:0,1666:0,1672:0,1673:0,1675:0,1676:0,1677:0,1678:0,1679:0,1680:0,1681:0,1682:0,1685:0,1689:0,1690:0,1691:0,1692:0,1693:0,1694:0,1695:0,1701:0,1702:0,1703:0,1704:0,1705:0,1706:0,1710:0,1711:0,1712:0,1713:0,1714:0,1718:0,1719:0,1720:0,1721:0,1722:0,1732:0,1733:0,1734:0,1739:0,1740:0,1741:0,1742:0,1743:0,1744:0,1748:0,1749:0,1750:0,1751:0,1752:0,1753:0,1754:0,1762:0,1763:0,1764:0,1765:0,1766:0,1777:0,1778:0,1784:0,1785:0,1786:0,1794:0,1795:0,1796:0,1797:0,1819:0,1798:0,1799:0,1800:0,1805:0,1806:0,1807:0,1808:0,1809:0,1810:0,1815:0,1816:0,1817:0,1818:0,1819:0,1820:1,0:1,2:1,3:1,10:1,11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:2,0:2,1:2,2:2,3:2,4:2,5:2,10:2,11:2,12:2,13:2,14:2,15:2,16:2,17:2,18:2,19:3,2:3,3:3,4:3,5:3,12:3,13:3,14:3,15:3,16:3,17:4,12:4,13:4,14:4,15:4,16:")
 ;	(integrator "hybrid_area_lights")
 )
